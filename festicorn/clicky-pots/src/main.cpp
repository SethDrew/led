#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <esp_task_wdt.h>
#include <clicky_packet_v1.h>

// Faroudja LD100 front panel → LED instrument. 6 push-pull pots + 6 buttons.
// Wiring: see festicorn/catalog/boards.yaml (role: clicky-pots).

// RF A/B probe: 0 = production; 1 = 20Hz broadcast + indicators forced
// all-on (max LED activity); 2 = 20Hz broadcast + indicators disabled;
// 3 = synthetic pot2 sweep, production cadence; 4 = synthetic sweep + 20Hz
#define RF_AB_TEST 0
#define IND_I2C_HZ 400000
// per-frame pilot-pixel trace for the first N ms after boot (0 = off);
// normal debug JSON is muted during the window
#define IND_TRACE_MS 0

const int NUM_POTS = 6;
const int WHITE_PINS[NUM_POTS] = {36, 39, 34, 35, 32, 33};
const int ADS_CH[NUM_POTS] = {3, 2, 1, 1, 2, 3};
const int NUM_BTNS = 7;
const int BUTTON_PINS[NUM_BTNS] = {25, 26, 14, 27, 16, 13, 17};

// Panel indicator pixels: WS2812B off the 5V bus (VIN), data 3.3V direct.
// Last pixel in the chain sits above the toggle; the first 12 spread over
// the other controls (not 1-1).
const int IND_PIN = 18;
const int IND_N = 13;
// LED 0 sits above the toggle (photo-confirmed strip direction 2026-08-23);
// if the pilot shows at the wrong end, this is the constant to flip back to 12
const int IND_TOGGLE = 12;
const int TOGGLE_BTN = 6;
const float IND_WARM_STEP = 0.12f;
const float IND_COOL_STEP = 0.06f;
// ignition flash: rust family borrowed from wormhole_eth (RUST 190,58,16)
const float IND_HOT[3] = {255, 95, 12};
// resting color: uniform burnt orange across the whole bar
const float IND_PAL[IND_N][3] = {
    {245, 100, 25}, {245, 100, 25}, {245, 100, 25}, {245, 100, 25},
    {245, 100, 25}, {245, 100, 25}, {245, 100, 25}, {245, 100, 25},
    {245, 100, 25}, {245, 100, 25}, {245, 100, 25}, {245, 100, 25},
    {245, 100, 25},
};

// preview mode: whole bar breathes between two colors, 10s each way
// (cosine, 20s period). set to 0 to return to the per-pixel IND_PAL.
#define IND_PREVIEW 0
const float IND_FADE_A[3] = {245, 100, 25};  // burnt orange
const float IND_FADE_B[3] = {190, 58, 16};   // wormhole rust (original)

const float IND_MASTER = 45.0f / 255.0f;
uint8_t indFrame[IND_N * 3];
bool indOn = false;
unsigned long indFlipMs = 0;
float indHeat[IND_N] = {0};
float indSettle[IND_N] = {0};
float indFlick = 0.0f;

// Adafruit's ESP32 show() streams via RMT with a 1-block buffer; ESP-NOW/WiFi
// interrupt latency starves the refill ISR mid-frame, truncating frames
// (visible strobe). Bit-bang is no good either at this board's 80MHz CPU.
// Fix: own RMT channel with 6 mem blocks — the whole 13-pixel frame (312 bit
// symbols) preloads into RMT RAM, hardware clocks it out, no refills.
#include "driver/rmt.h"

void indRmtInit() {
    rmt_config_t cfg = RMT_DEFAULT_CONFIG_TX((gpio_num_t)IND_PIN, RMT_CHANNEL_0);
    cfg.clk_div = 2;
    cfg.mem_block_num = 6;
    rmt_config(&cfg);
    rmt_driver_install(RMT_CHANNEL_0, 0, 0);
}

void indShow() {
    static uint8_t prev[IND_N * 3];
    static bool first = true;
    static rmt_item32_t items[IND_N * 24];
    const uint8_t *px = indFrame;
    const int nb = IND_N * 3;
    if (!first && memcmp(prev, px, nb) == 0) return;
    rmt_wait_tx_done(RMT_CHANNEL_0, pdMS_TO_TICKS(10));
    memcpy(prev, px, nb);
    first = false;
    int k = 0;
    for (int i = 0; i < nb; i++) {
        for (int b = 7; b >= 0; b--) {
            bool one = (px[i] >> b) & 1;
            items[k].level0 = 1;
            items[k].duration0 = one ? 32 : 16;
            items[k].level1 = 0;
            items[k].duration1 = one ? 18 : 34;
            k++;
        }
    }
    rmt_write_items(RMT_CHANNEL_0, items, k, false);
}

void renderIndicators(unsigned long now, float dt) {
    float p = (now - indFlipMs) * 0.001f;
    float t = now * 0.001f;
    indFlick += (random(1000) * 0.001f - 0.5f) * dt * 10.0f;
    indFlick *= expf(-3.0f * dt);
    for (int i = 0; i < IND_N; i++) {
        bool lit = indOn ? (IND_TOGGLE - i) * IND_WARM_STEP <= p
                         : i * IND_COOL_STEP > p;
        float target = lit ? 1.0f : 0.0f;
        if (!indOn && !lit && i >= IND_TOGGLE - 1) {
            // trough floor keeps red >=4 codes / green >=1.7 — below that the
            // 0-snap and the LED's coarse low-end quanta step visibly
            target = 0.42f + 0.12f * sinf(t * 6.2832f / 4.5f) + indFlick * 0.08f;
            if (target < 0.30f) target = 0.30f;
        }
        float tauH = (target > indHeat[i]) ? 0.05f : 0.12f;
        indHeat[i] += (target - indHeat[i]) * fminf(1.0f, dt / tauH);
        float sTgt = (indOn && lit) ? 1.0f : 0.0f;
        float tauS = (sTgt > indSettle[i]) ? 1.2f : 0.15f;
        indSettle[i] += (sTgt - indSettle[i]) * fminf(1.0f, dt / tauS);
        float s = indSettle[i];
        float lvl = indHeat[i] * (1.0f - 0.2f * s);
        float l2 = lvl * lvl;
        // float -> GRB wire order with delta-sigma dither; snap the 0<->1
        // code boundary to 0 (dithering across it flashes visibly)
        static const int ord[3] = {1, 0, 2};
#if IND_PREVIEW
        float mix = 0.5f - 0.5f * cosf(t * 6.2832f / 20.0f);
        float fade[3] = {IND_FADE_A[0] + (IND_FADE_B[0] - IND_FADE_A[0]) * mix,
                         IND_FADE_A[1] + (IND_FADE_B[1] - IND_FADE_A[1]) * mix,
                         IND_FADE_A[2] + (IND_FADE_B[2] - IND_FADE_A[2]) * mix};
        const float *pal = fade;
#else
        const float *pal = IND_PAL[i];
#endif
        for (int c = 0; c < 3; c++) {
            int j = i * 3 + c;
            int oc = ord[c];
            float tgt = (IND_HOT[oc] + (pal[oc] - IND_HOT[oc]) * s)
                        * l2 * IND_MASTER;
            if (tgt < 1.0f) {
                indFrame[j] = 0;
                continue;
            }
            // no dithering: plain rounded codes, 0.6-code hysteresis so
            // noise can't chatter a boundary
            if (fabsf(tgt - (float)indFrame[j]) > 0.6f) {
                int o = (int)(tgt + 0.5f);
                if (o > 255) o = 255;
                indFrame[j] = (uint8_t)o;
            }
        }
    }
    indShow();
}

const float RREF = 330.0f;
const float VCC = 3.3f;

// Pot idx 3 is mirror-assembled: red taper reversed, pull-switch inverted
// (its sense line is open when pressed, follows the element when pulled).
const int MIRROR_POT = 3;

// Dual calibration: the push-pull switch reroutes the pot network, so the
// red channel has a DIFFERENT resistance curve pressed-in vs pulled-out.
// [0][i] = pressed, [1][i] = pulled. Placeholder ranges — bake in real
// values from a calibration recording (tools/record_serial.py sweep).
// Fitted from recordings/tuning_session.jsonl: pressed = end-stop ohms
// (0.3/99.7 pct); pulled = least-squares over no-rotation pull/push flips
// (position continuity) + pulled end stops. Refit with the same procedure
// if a knob develops a jump at pull/push.
float potCalMin[2][NUM_POTS] = {
    {1, 1, 1, 1, 1, 1},
    {2, 1, 1, 1, 0, 1},
};
float potCalMax[2][NUM_POTS] = {
    {887, 811, 876, 497, 842, 829},
    {578, 526, 577, 998, 564, 528},
};

// Transition rejection (see fiddle_session recording): white is strictly
// bimodal at rest (pressed >=1.44V, pulled ~0V); anything between means the
// switch is mechanically in flight and ALL reads are garbage — hold output.
// Mirror pot: pressed = open (3.3V), pulled = element voltage (<2.5V).
const float JUMP_THRESH = 0.15f;
const float CONFIRM_TOL = 0.06f;
float pendingP[NUM_POTS] = {0};
bool pendingValid[NUM_POTS] = {false};

Adafruit_ADS1115 adsA;
Adafruit_ADS1115 adsB;
bool adsAOk = false, adsBOk = false;
uint32_t adsFailCnt = 0, adsReinit = 0;
uint32_t potStatDiscard = 0, potStatMismatch = 0, potStatPend = 0,
         potStatCommit = 0;

float potPos[NUM_POTS] = {0};
// Transmitted position: hysteresis latch over potPos. ADC quantization dither
// (~0.002 at a code boundary) otherwise rides out in the full-state packets
// that button edges trigger, visibly wiggling a particle parked on the edge.
float potPosTx[NUM_POTS] = {0};
const float TX_HYST = 0.002f;
bool potPulled[NUM_POTS] = {false};
// The pull switch bounces for tens of ms; at 66Hz/pot sampling the bounce is
// visible (the old 28Hz loop was an accidental debounce). Commit a pull-state
// change only after 8 consecutive agreeing samples (~20ms).
uint8_t pullRun[NUM_POTS] = {0};
int16_t potRaw[NUM_POTS] = {0};
int whiteRaw[NUM_POTS] = {0};

bool btnState[NUM_BTNS] = {false};
bool btnLast[NUM_BTNS] = {false};
unsigned long btnDebounceTime[NUM_BTNS] = {0};
const unsigned long DEBOUNCE_MS = 20;

uint32_t espnowSeq = 0;
uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
volatile uint32_t txQueued = 0, txQueueFail = 0, txCbOk = 0, txCbFail = 0;
volatile uint32_t txCbMaxLat = 0, txLastSendMs = 0;

void onEspnowSendCb(const uint8_t *mac, esp_now_send_status_t st) {
    uint32_t lat = millis() - txLastSendMs;
    if (lat > txCbMaxLat) txCbMaxLat = lat;
    if (st == ESP_NOW_SEND_SUCCESS) txCbOk++; else txCbFail++;
}
unsigned long lastBroadcast = 0;
const unsigned long HEARTBEAT_MS =
    (RF_AB_TEST == 0 || RF_AB_TEST == 3) ? 1000 : 50;

float prevPos[NUM_POTS] = {-1};
bool prevPulled[NUM_POTS] = {false};
bool prevBtn[NUM_BTNS] = {false};
const float POS_EPSILON = 0.004f;

float potRawToR(int16_t raw) {
    float v = raw * 0.000125f;
    if (v >= VCC - 0.05f) return 1e6f;
    if (v <= 0.001f) return 0.0f;
    return v * RREF / (VCC - v);
}

static const uint16_t ADS_MUX[4] = {
    ADS1X15_REG_CONFIG_MUX_SINGLE_0, ADS1X15_REG_CONFIG_MUX_SINGLE_1,
    ADS1X15_REG_CONFIG_MUX_SINGLE_2, ADS1X15_REG_CONFIG_MUX_SINGLE_3};

// Non-blocking ADS scan: each chip has one conversion in flight; the loop
// starts it, keeps rendering, and harvests when the chip signals done. Both
// chips convert in parallel — ~60Hz per pot without ever blocking the loop.
// 860SPS converts in ~1.2ms; 5ms of no completion means the bus is gone
// (the 12h silent-hang bug lived in the old unbounded poll).
struct AdsScan {
    Adafruit_ADS1115 *ads;
    bool *ok;
    int base;
    int ch;
    bool inFlight;
    int s1;
    unsigned long startMs;
};
AdsScan adsScan[2] = {{&adsA, &adsAOk, 0, 0, false, 0, 0},
                      {&adsB, &adsBOk, 3, 0, false, 0, 0}};

// 9 SCL pulses release a slave that's mid-transaction holding SDA low.
void i2cBusRecover() {
    pinMode(22, OUTPUT);
    pinMode(21, INPUT_PULLUP);
    for (int i = 0; i < 9 && digitalRead(21) == LOW; i++) {
        digitalWrite(22, LOW);  delayMicroseconds(5);
        digitalWrite(22, HIGH); delayMicroseconds(5);
    }
}

float potRawToPos(int i, int16_t raw, bool pulled) {
    int s = pulled ? 1 : 0;
    float r = potRawToR(raw);
    float p = (r - potCalMin[s][i]) / (potCalMax[s][i] - potCalMin[s][i]);
    p = constrain(p, 0.0f, 1.0f);
    return (i == MIRROR_POT) ? 1.0f - p : p;
}

// 0 = pressed, 1 = pulled, -1 = in-flight (white in the forbidden band)
int classifyWhite(int i) {
    whiteRaw[i] = analogRead(WHITE_PINS[i]);
    float v = whiteRaw[i] * VCC / 4095.0f;
    if (i == MIRROR_POT) {
        if (v > 3.1f) return 0;
        if (v < 2.5f) return 1;
        return -1;
    }
    if (v < 0.15f) return 1;
    if (v > 1.2f) return 0;
    return -1;
}

void potScanStep() {
    for (int c = 0; c < 2; c++) {
        AdsScan &sc = adsScan[c];
        if (!*sc.ok) {
            sc.inFlight = false;
            continue;
        }
        int i = sc.base + sc.ch;
        if (!sc.inFlight) {
            sc.s1 = classifyWhite(i);
            if (sc.s1 < 0) {
                potStatDiscard++;
                sc.ch = (sc.ch + 1) % 3;
                continue;
            }
            sc.ads->startADCReading(ADS_MUX[ADS_CH[i]], false);
            sc.startMs = millis();
            sc.inFlight = true;
        } else if (sc.ads->conversionComplete()) {
            sc.inFlight = false;
            int16_t raw = sc.ads->getLastConversionResults();
            int s2 = classifyWhite(i);
            sc.ch = (sc.ch + 1) % 3;
            if (s2 != sc.s1) {
                potStatMismatch++;
                continue;
            }
            potRaw[i] = raw;
            bool pulledNow = (sc.s1 == 1);
            if (pulledNow != potPulled[i]) {
                if (++pullRun[i] >= 8) {
                    potPulled[i] = pulledNow;
                    pullRun[i] = 0;
                }
            } else {
                pullRun[i] = 0;
            }
            float p = potRawToPos(i, raw, pulledNow);
            if (fabsf(p - potPos[i]) > JUMP_THRESH) {
                if (pendingValid[i] && fabsf(p - pendingP[i]) < CONFIRM_TOL) {
                    potPos[i] = p;
                    pendingValid[i] = false;
                    potStatCommit++;
                } else {
                    pendingP[i] = p;
                    pendingValid[i] = true;
                    potStatPend++;
                }
            } else {
                potPos[i] = p;
                pendingValid[i] = false;
                potStatCommit++;
            }
        } else if (millis() - sc.startMs > 5) {
            *sc.ok = false;
            sc.inFlight = false;
            adsFailCnt++;
        }
    }
}

// All I2C (scan + bus recovery + re-init) lives on this core-0 task; the
// core-1 loop only reads the shared word-atomic pot state.
void potScanTask(void *) {
    unsigned long lastRetry = 0;
    for (;;) {
        if ((!adsAOk || !adsBOk) && millis() - lastRetry >= 1000) {
            lastRetry = millis();
            adsReinit++;
            i2cBusRecover();
            Wire.begin(21, 22);
            Wire.setClock(IND_I2C_HZ);
            if (!adsAOk) adsAOk = adsA.begin(0x48, &Wire);
            if (!adsBOk) adsBOk = adsB.begin(0x49, &Wire);
        }
        potScanStep();
        potScanStep();
        vTaskDelay(1);
    }
}

void broadcastState() {
    ClickyPacketV1 pkt = {};
    pkt.magic = CLICKY_MAGIC;
    pkt.ver = CLICKY_VER;
    pkt.seq = (uint8_t)(espnowSeq++);
    for (int i = 0; i < NUM_POTS; i++) {
        pkt.pos[i] = (uint16_t)(potPosTx[i] * 10000.0f + 0.5f);
        if (potPulled[i]) pkt.pullBits |= (1 << i);
    }
    for (int i = 0; i < NUM_BTNS; i++)
        if (btnState[i]) pkt.btnBits |= (1 << i);
    txLastSendMs = millis();
    if (esp_now_send(broadcastAddr, (uint8_t *)&pkt, sizeof(pkt)) == ESP_OK)
        txQueued++;
    else
        txQueueFail++;
}

void setup() {
    setCpuFrequencyMhz(80);
    Serial.begin(230400);

    for (int i = 0; i < NUM_BTNS; i++) pinMode(BUTTON_PINS[i], INPUT_PULLUP);
    analogSetAttenuation(ADC_11db);

    indRmtInit();
    indShow();

    i2cBusRecover();
    Wire.begin(21, 22);
    Wire.setClock(IND_I2C_HZ);
    Wire.setTimeOut(20);
    adsA.setGain(GAIN_ONE);
    adsB.setGain(GAIN_ONE);
    adsA.setDataRate(RATE_ADS1115_860SPS);
    adsB.setDataRate(RATE_ADS1115_860SPS);
    adsAOk = adsA.begin(0x48, &Wire);
    adsBOk = adsB.begin(0x49, &Wire);
    Serial.printf("ADS1115 A(0x48): %s  B(0x49): %s\n",
                  adsAOk ? "OK" : "FAIL", adsBOk ? "OK" : "FAIL");
    xTaskCreatePinnedToCore(potScanTask, "potscan", 4096, NULL, 1, NULL, 0);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    if (esp_now_init() == ESP_OK) {
        Serial.println("ESP-NOW OK ch1");
        esp_now_register_send_cb(onEspnowSendCb);
        // wake_window 0 = radio sleeps except during TX; this board can never receive
        esp_wifi_set_ps(WIFI_PS_MIN_MODEM);
        esp_now_set_wake_window(0);
        esp_now_peer_info_t peer = {};
        memcpy(peer.peer_addr, broadcastAddr, 6);
        peer.channel = 0;
        peer.encrypt = false;
        esp_now_add_peer(&peer);
    } else {
        Serial.println("ESP-NOW FAIL");
    }

    // Core already panic-reboots on idle-CPU0 starvation; subscribing the loop
    // task makes the WDT also catch a loop that stalls while yielding.
    esp_task_wdt_add(NULL);
}

void loop() {
    unsigned long lp0 = millis();
    static unsigned long prevLp0 = 0, maxLoopGap = 0;
    if (prevLp0 && lp0 - prevLp0 > maxLoopGap) maxLoopGap = lp0 - prevLp0;
    prevLp0 = lp0;
    esp_task_wdt_reset();

    unsigned long lpPot = millis();

    for (int i = 0; i < NUM_BTNS; i++) {
        // Momentaries (0-5) are normally-closed: rest = 0Ω to GND = LOW.
        // Pressed = open = HIGH. Toggle (6): open = HIGH = on (flipped
        // 2026-08-23 to match the panel's preferred throw direction).
        bool raw = digitalRead(BUTTON_PINS[i]) == HIGH;
        if (raw != btnLast[i]) { btnLast[i] = raw; btnDebounceTime[i] = millis(); }
        if (millis() - btnDebounceTime[i] > DEBOUNCE_MS) btnState[i] = btnLast[i];
    }

    static unsigned long indLastMs = 0;
    unsigned long indNow = millis();
    float indDt = (indNow - indLastMs) * 0.001f;
    if (indDt > 0.05f) indDt = 0.05f;
    indLastMs = indNow;
#if RF_AB_TEST == 1
    bool indWant = true;
#elif RF_AB_TEST == 2
    bool indWant = false;
#elif RF_AB_TEST >= 3
    bool indWant = true;
#else
    bool indWant = btnState[TOGGLE_BTN];
#endif
    if (indWant != indOn) {
        indOn = indWant;
        indFlipMs = indNow;
    }
#if RF_AB_TEST != 2
    renderIndicators(indNow, indDt);
#endif
#if IND_TRACE_MS > 0
    if (millis() < IND_TRACE_MS) {
        static uint32_t lastUs = 0;
        uint32_t nowUs = micros();
        Serial.printf("T,%lu,%u,%u,%u,%.4f\n",
                      (unsigned long)(nowUs - lastUs),
                      indFrame[IND_TOGGLE * 3 + 1], indFrame[IND_TOGGLE * 3],
                      indFrame[IND_TOGGLE * 3 + 2],
                      (double)indHeat[IND_TOGGLE]);
        lastUs = nowUs;
    }
#endif
    unsigned long lpInd = millis();

    for (int i = 0; i < NUM_POTS; i++)
        if (fabsf(potPos[i] - potPosTx[i]) > TX_HYST) potPosTx[i] = potPos[i];

#if RF_AB_TEST >= 3
    {
        float ph = fmodf(millis() * 0.000125f, 1.0f);
        potPosTx[2] = ph < 0.5f ? ph * 2.0f : 2.0f - ph * 2.0f;
    }
#endif

    bool changed = false;
    for (int i = 0; i < NUM_POTS; i++) {
        if (fabsf(potPosTx[i] - prevPos[i]) > POS_EPSILON ||
            potPulled[i] != prevPulled[i]) {
            changed = true;
            break;
        }
    }
    for (int i = 0; i < NUM_BTNS; i++)
        if (btnState[i] != prevBtn[i]) changed = true;

    if (changed || millis() - lastBroadcast >= HEARTBEAT_MS) {
        broadcastState();
        lastBroadcast = millis();
        for (int i = 0; i < NUM_POTS; i++) {
            prevPos[i] = potPosTx[i];
            prevPulled[i] = potPulled[i];
        }
        for (int i = 0; i < NUM_BTNS; i++) prevBtn[i] = btnState[i];
    }
    unsigned long lpTx = millis();

    static unsigned long lastDebug = 0;
    if (millis() - lastDebug >= 50 && millis() >= IND_TRACE_MS) {
        lastDebug = millis();
        Serial.printf("{\"ms\":%lu,\"p\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f],"
                      "\"pull\":[%d,%d,%d,%d,%d,%d],\"btn\":[%d,%d,%d,%d,%d,%d,%d],"
                      "\"raw\":[%d,%d,%d,%d,%d,%d],\"wraw\":[%d,%d,%d,%d,%d,%d],"
                      "\"txq\":[%u,%u,%u,%u,%u],\"gap\":%lu,"
                      "\"adsf\":%u,\"adsr\":%u,\"adsok\":%d,"
                      "\"pf\":[%u,%u,%u,%u]}\n",
                      millis(),
                      potPos[0], potPos[1], potPos[2], potPos[3], potPos[4], potPos[5],
                      potPulled[0], potPulled[1], potPulled[2],
                      potPulled[3], potPulled[4], potPulled[5],
                      btnState[0], btnState[1], btnState[2],
                      btnState[3], btnState[4], btnState[5], btnState[6],
                      potRaw[0], potRaw[1], potRaw[2], potRaw[3], potRaw[4], potRaw[5],
                      whiteRaw[0], whiteRaw[1], whiteRaw[2],
                      whiteRaw[3], whiteRaw[4], whiteRaw[5],
                      (unsigned)txQueued, (unsigned)txQueueFail,
                      (unsigned)txCbOk, (unsigned)txCbFail,
                      (unsigned)txCbMaxLat, maxLoopGap,
                      (unsigned)adsFailCnt, (unsigned)adsReinit,
                      (adsAOk ? 1 : 0) + (adsBOk ? 2 : 0),
                      (unsigned)potStatDiscard, (unsigned)potStatMismatch,
                      (unsigned)potStatPend, (unsigned)potStatCommit);
        txCbMaxLat = 0;
        maxLoopGap = 0;
    }

    unsigned long lpEnd = millis();
    if (lpEnd - lp0 > 60)
        Serial.printf("{\"stall\":%lu,\"pot\":%lu,\"ind\":%lu,\"tx\":%lu,\"dbg\":%lu}\n",
                      lpEnd - lp0, lpPot - lp0, lpInd - lpPot,
                      lpTx - lpInd, lpEnd - lpTx);

    delay(1);
}
