#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <clicky_packet_v1.h>

// Faroudja LD100 front panel → LED instrument. 6 push-pull pots + 6 buttons.
// Wiring: see festicorn/catalog/boards.yaml (role: clicky-pots).

const int NUM_POTS = 6;
const int WHITE_PINS[NUM_POTS] = {36, 39, 34, 35, 32, 33};
const int ADS_CH[NUM_POTS] = {3, 2, 1, 1, 2, 3};
const int NUM_BTNS = 7;
const int BUTTON_PINS[NUM_BTNS] = {25, 26, 14, 27, 16, 13, 17};

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

float potPos[NUM_POTS] = {0};
// Transmitted position: hysteresis latch over potPos. ADC quantization dither
// (~0.002 at a code boundary) otherwise rides out in the full-state packets
// that button edges trigger, visibly wiggling a particle parked on the edge.
float potPosTx[NUM_POTS] = {0};
const float TX_HYST = 0.002f;
bool potPulled[NUM_POTS] = {false};
int16_t potRaw[NUM_POTS] = {0};
int whiteRaw[NUM_POTS] = {0};

bool btnState[NUM_BTNS] = {false};
bool btnLast[NUM_BTNS] = {false};
unsigned long btnDebounceTime[NUM_BTNS] = {0};
const unsigned long DEBOUNCE_MS = 20;

uint32_t espnowSeq = 0;
uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
unsigned long lastBroadcast = 0;
const unsigned long HEARTBEAT_MS = 1000;

float prevPos[NUM_POTS] = {-1};
bool prevPulled[NUM_POTS] = {false};
bool prevBtn[NUM_BTNS] = {false};
const float POS_EPSILON = 0.01f;

float potRawToR(int16_t raw) {
    float v = raw * 0.000125f;
    if (v >= VCC - 0.05f) return 1e6f;
    if (v <= 0.001f) return 0.0f;
    return v * RREF / (VCC - v);
}

float readPotPosition(int i, bool pulled) {
    int16_t raw = (i < 3)
        ? (adsAOk ? adsA.readADC_SingleEnded(ADS_CH[i]) : 0)
        : (adsBOk ? adsB.readADC_SingleEnded(ADS_CH[i]) : 0);
    potRaw[i] = raw;
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
    esp_now_send(broadcastAddr, (uint8_t *)&pkt, sizeof(pkt));
}

void setup() {
    Serial.begin(230400);

    for (int i = 0; i < NUM_BTNS; i++) pinMode(BUTTON_PINS[i], INPUT_PULLUP);
    analogSetAttenuation(ADC_11db);

    Wire.begin(21, 22);
    adsA.setGain(GAIN_ONE);
    adsB.setGain(GAIN_ONE);
    adsA.setDataRate(RATE_ADS1115_860SPS);
    adsB.setDataRate(RATE_ADS1115_860SPS);
    adsAOk = adsA.begin(0x48, &Wire);
    adsBOk = adsB.begin(0x49, &Wire);
    Serial.printf("ADS1115 A(0x48): %s  B(0x49): %s\n",
                  adsAOk ? "OK" : "FAIL", adsBOk ? "OK" : "FAIL");

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    if (esp_now_init() == ESP_OK) {
        Serial.println("ESP-NOW OK ch1");
        esp_now_peer_info_t peer = {};
        memcpy(peer.peer_addr, broadcastAddr, 6);
        peer.channel = 0;
        peer.encrypt = false;
        esp_now_add_peer(&peer);
    } else {
        Serial.println("ESP-NOW FAIL");
    }
}

void loop() {
    for (int i = 0; i < NUM_POTS; i++) {
        int s1 = classifyWhite(i);
        if (s1 >= 0) {
            float p = readPotPosition(i, s1 == 1);
            int s2 = classifyWhite(i);
            if (s2 == s1) {
                potPulled[i] = (s1 == 1);
                if (fabsf(p - potPos[i]) > JUMP_THRESH) {
                    if (pendingValid[i] && fabsf(p - pendingP[i]) < CONFIRM_TOL) {
                        potPos[i] = p;
                        pendingValid[i] = false;
                    } else {
                        pendingP[i] = p;
                        pendingValid[i] = true;
                    }
                } else {
                    potPos[i] = p;
                    pendingValid[i] = false;
                }
            }
        }

    }

    for (int i = 0; i < NUM_BTNS; i++) {
        // Momentaries (0-5) are normally-closed: rest = 0Ω to GND = LOW.
        // Pressed = open = HIGH. Toggle (6) is plain: closed = LOW = on.
        bool raw = digitalRead(BUTTON_PINS[i]) == (i < 6 ? HIGH : LOW);
        if (raw != btnLast[i]) { btnLast[i] = raw; btnDebounceTime[i] = millis(); }
        if (millis() - btnDebounceTime[i] > DEBOUNCE_MS) btnState[i] = btnLast[i];
    }

    for (int i = 0; i < NUM_POTS; i++)
        if (fabsf(potPos[i] - potPosTx[i]) > TX_HYST) potPosTx[i] = potPos[i];

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

    static unsigned long lastDebug = 0;
    if (millis() - lastDebug >= 10) {
        lastDebug = millis();
        Serial.printf("{\"ms\":%lu,\"p\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f],"
                      "\"pull\":[%d,%d,%d,%d,%d,%d],\"btn\":[%d,%d,%d,%d,%d,%d,%d],"
                      "\"raw\":[%d,%d,%d,%d,%d,%d],\"wraw\":[%d,%d,%d,%d,%d,%d]}\n",
                      millis(),
                      potPos[0], potPos[1], potPos[2], potPos[3], potPos[4], potPos[5],
                      potPulled[0], potPulled[1], potPulled[2],
                      potPulled[3], potPulled[4], potPulled[5],
                      btnState[0], btnState[1], btnState[2],
                      btnState[3], btnState[4], btnState[5], btnState[6],
                      potRaw[0], potRaw[1], potRaw[2], potRaw[3], potRaw[4], potRaw[5],
                      whiteRaw[0], whiteRaw[1], whiteRaw[2],
                      whiteRaw[3], whiteRaw[4], whiteRaw[5]);
    }

    delay(2);
}
