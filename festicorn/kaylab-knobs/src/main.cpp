#include <Arduino.h>
#include <driver/adc.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <Preferences.h>

Preferences prefs;
uint32_t bootCount = 0;

// Meters: 500µA full-scale, 5.6kΩ series resistor
// µA meter: DAC 230 = 500µA full-scale
// V meter:  DAC 245 = 500V full-scale (171.9Ω coil)
const int METER_UA_PIN = 25; // DAC1
const int METER_V_PIN  = 26; // DAC2
int METER_UA_MAX = 230;
int METER_V_MAX  = 245;

// DC switch: voltage divider sense, HIGH=ON, LOW=OFF
const int DC_SWITCH_PIN = 14;
// AC switch: ~100Ω closed (oxidized), open when ON
const int AC_SWITCH_PIN = 13;
// Hundreds: 5-position rotary switch, 10kΩ pullup to 3.3V
const int HUNDREDS_PIN = 35;
// Tens: 10MΩ knob, 10 positions, 5.6MΩ pullup to 3.3V, cap to GND
const int TENS_PIN = 32;
// Ones: 1MΩ knob, 11 positions (labeled 1.02, 2–11), 1MΩ pullup to 3.3V
const int ONES_PIN = 34;
int hundredsPos = 0;
int tensPos = 0;
int onesPos = 0;

// Decade knob bank — ESP32 ADC fallback (until ADS1115 works)
// Ones (white-green): 510Ω pullup to 3.3V, 1µF cap, 0-8.9Ω
const int DECADE_ONES_PIN = 39;  // VN
// Tens+hundreds (blue-green): 1kΩ pullup to 3.3V, 1µF cap, 0-1099Ω
const int DECADE_TH_PIN = 36;    // VP

int uaDac = 0;
int vDac  = 0;

// Meters mirror tree-of-record state by default: V=brightness (tens),
// uA=speed (hundreds). An explicit set_ua_dac/set_v_dac command takes
// over that meter until the driving knob next moves.
bool uaAuto = true;
bool vAuto  = true;

// ADS1115 on I2C: SDA=GPIO23, SCL=GPIO22
Adafruit_ADS1115 ads;
bool adsOk = false;
int16_t adsOnesRaw = 0;
int16_t adsTensRaw = 0;
float adsOnesV = 0;
float adsTensV = 0;

// Decade position decoding
int decOuter = 0, decMid = 0, decInner = 0;
int decValue = 0;
int decCandValue = -1;
int decConfirm = 0;

const float PULLUP_ONES = 510.0;
const float VCC = 3.394;
const float ADS_LSB_GAIN2 = 0.0000625;

const int16_t hundredsLUT[13] = {6, 2440, 4456, 6171, 7631, 8903, 10009, 10991, 11857, 12637, 13334, 13968, 14546};
const int16_t hundredsThresh[11] = {2326, 4366, 6090, 7565, 8842, 9958, 10943, 11817, 12599, 13302, 13937};

float onesRawToR(int16_t raw) {
    float v = raw * ADS_LSB_GAIN2;
    if (v >= VCC - 0.01) return 9.0;
    if (v <= 0.001) return 0.0;
    return v * PULLUP_ONES / (VCC - v);
}

// Debounce state for switches
bool dcOn = false, acOn = false;
bool dcLast = false, acLast = false;
unsigned long dcDebounceTime = 0, acDebounceTime = 0;
const unsigned long DEBOUNCE_MS = 100;

// ESP-NOW broadcast state
uint32_t espnowSeq = 0;
uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
unsigned long lastBroadcast = 0;
const unsigned long HEARTBEAT_MS = 1000;

// Native-ADC knob noise rejection: hysteresis margin (counts a reading must
// cross a neighboring threshold by before switching bins) and consecutive-read
// confirm count. Both filter the high-impedance ADC jitter that worsens when
// the controller loses a clean USB ground reference.
const int KNOB_HYST = 40;
const int KNOB_CONFIRM = 3;
bool stateChanged = false;

// Previous state for change detection
bool prevDcOn = false, prevAcOn = false;
int prevHundredsPos = 0, prevTensPos = 0, prevOnesPos = 0;
int prevDecValue = 0, prevUaDac = 0, prevVDac = 0;
int prevUaMax = 0, prevVMax = 0;

String serialBuf = "";

// N-sample debounce for a quantized knob position. A candidate must be read
// `need` times in a row before it commits to `out`, so brief ADC jitter never
// propagates. Each knob owns its own pending/count state. Returns true on change.
bool debouncePos(int cand, int &out, int &pending, int &pendingCount, int need) {
    if (cand == out) { pending = -1; pendingCount = 0; return false; }
    if (cand == pending) {
        if (++pendingCount >= need) {
            out = cand;
            pending = -1;
            pendingCount = 0;
            return true;
        }
    } else {
        pending = cand;
        pendingCount = 1;
    }
    return false;
}

// Decode a raw ADC value into a bin using a DESCENDING threshold ladder, with
// hysteresis: the reading must cross a neighboring threshold by `margin` counts
// before leaving the current bin, killing chatter when the knob sits on a
// boundary. thresh[k] is the lower bound of bin k; bin count is nThresh+1.
//   raw > thresh[0]         → bin 0
//   thresh[1] < raw <= [0]  → bin 1  ... raw <= thresh[last] → bin nThresh
int decodeBinHyst(int raw, int cur, const int *thresh, int nThresh, int margin) {
    // Plain candidate (no hysteresis).
    int cand = nThresh;
    for (int k = 0; k < nThresh; k++) {
        if (raw > thresh[k]) { cand = k; break; }
    }
    if (cand == cur) return cur;
    // Moving to a LOWER bin index (raw rose): require raw above the boundary
    // between cur and cur-1 (thresh[cur-1]) by +margin.
    if (cand < cur) {
        if (cur - 1 >= 0 && raw <= thresh[cur - 1] + margin) return cur;
    } else { // moving to a HIGHER bin index (raw fell)
        if (cur < nThresh && raw > thresh[cur] - margin) return cur;
    }
    return cand;
}

int parseIntField(const String &line, const char *key) {
    int idx = line.indexOf(key);
    if (idx < 0) return -1;
    int c = line.indexOf(':', idx);
    if (c < 0) return -1;
    return line.substring(c + 1).toInt();
}

void handleCommand(const String &line) {
    int v;
    if ((v = parseIntField(line, "\"set_ua_max\"")) >= 0 && v <= 255) METER_UA_MAX = v;
    if ((v = parseIntField(line, "\"set_v_max\""))  >= 0 && v <= 255) METER_V_MAX  = v;
    if ((v = parseIntField(line, "\"set_ua_dac\"")) >= 0 && v <= 255) { uaDac = v; uaAuto = false; }
    if ((v = parseIntField(line, "\"set_v_dac\""))  >= 0 && v <= 255) { vDac  = v; vAuto  = false; }
}

void onEspNowRecv(const uint8_t *mac, const uint8_t *data, int len) {
    String msg = "";
    for (int i = 0; i < len && i < 128; i++) msg += (char)data[i];
    handleCommand(msg);
}

static int lastHundredsRaw = 0, lastTensRaw = 0, lastOnesRaw = 0;
static int16_t lastAdsTensRaw = 0, lastAdsOnesRaw = 0, lastAdsVccRef = 0;
// ADS Vcc ref reading when thresholds were calibrated (USB-grounded, 7.5k/7.5k divider)
static const float adsVccRefBaseline = 13249.0f;

void broadcastState() {
    char buf[250];
    int n = snprintf(buf, sizeof(buf),
        "{\"dc\":%d,\"ac\":%d,\"h\":%d,\"t\":%d,\"o\":%d,"
        "\"dec\":%d,\"do\":%d,\"dm\":%d,\"di\":%d,"
        "\"ua\":%d,\"v\":%d,"
        "\"rh\":%d,\"rt\":%d,\"ro\":%d,\"at\":%d,\"ao\":%d,\"ar\":%d,"
        "\"up\":%lu,\"b\":%u,\"s\":%u}",
        (int)dcOn, (int)acOn,
        hundredsPos, tensPos, onesPos, decValue, decOuter, decMid, decInner,
        uaDac, vDac,
        lastHundredsRaw, lastTensRaw, lastOnesRaw,
        (int)lastAdsTensRaw, (int)lastAdsOnesRaw, (int)lastAdsVccRef,
        millis(), bootCount, espnowSeq++);
    if (n > 250) n = 250;
    esp_now_send(broadcastAddr, (uint8_t *)buf, n);
}

void setup() {
    Serial.begin(115200);

    prefs.begin("bs26", false);
    bootCount = prefs.getUInt("boots", 0) + 1;
    prefs.putUInt("boots", bootCount);
    prefs.end();

    pinMode(DC_SWITCH_PIN, INPUT);
    pinMode(AC_SWITCH_PIN, INPUT);

    analogSetAttenuation(ADC_11db);
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_DB_11);  // GPIO 36 VP
    adc1_config_channel_atten(ADC1_CHANNEL_3, ADC_ATTEN_DB_11);  // GPIO 39 VN

    Wire.begin(23, 22);  // SDA=GPIO23, SCL=GPIO22
    ads.setGain(GAIN_ONE); // ±4.096V (0.125 mV/bit)
    adsOk = ads.begin(0x48, &Wire);
    if (adsOk) Serial.println("ADS1115 OK");
    else Serial.println("ADS1115 FAIL");

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);
    if (esp_now_init() == ESP_OK) {
        Serial.println("ESP-NOW OK ch1");
        esp_now_register_recv_cb(onEspNowRecv);
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
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialBuf.length()) { handleCommand(serialBuf); serialBuf = ""; }
        } else if (serialBuf.length() < 128) {
            serialBuf += c;
        }
    }

    bool dcRead = digitalRead(DC_SWITCH_PIN) == HIGH;
    bool acRead = digitalRead(AC_SWITCH_PIN) == HIGH;
    if (dcRead != dcLast) { dcLast = dcRead; dcDebounceTime = millis(); }
    if (acRead != acLast) { acLast = acRead; acDebounceTime = millis(); }
    if (millis() - dcDebounceTime > DEBOUNCE_MS) dcOn = dcLast;
    if (millis() - acDebounceTime > DEBOUNCE_MS) acOn = acLast;
    int hundredsRaw = analogRead(HUNDREDS_PIN);
    int tensRaw = analogRead(TENS_PIN);
    int onesRaw = analogRead(ONES_PIN);
    lastHundredsRaw = hundredsRaw;
    lastTensRaw = tensRaw;
    lastOnesRaw = onesRaw;
    int decInnerRaw = analogRead(DECADE_ONES_PIN);
    int decThRaw = analogRead(DECADE_TH_PIN);
    // Debug: try reading VP/VN by ADC channel directly
    int vp_adc = adc1_get_raw(ADC1_CHANNEL_0);  // GPIO 36
    int vn_adc = adc1_get_raw(ADC1_CHANNEL_3);  // GPIO 39

    // Tens: 10-pos, 5.6MΩ pullup, cap filtered, GPIO 32.
    // Unified thresholds calibrated for both USB-grounded and ungrounded operation.
    static const int TENS_THRESH[9] =
        { 3133, 2095, 1942, 1767, 1565, 1304, 995, 602, 197 };
    int tensCand = decodeBinHyst(tensRaw, tensPos, TENS_THRESH, 9, KNOB_HYST);
    static int tensPending = -1, tensPendingCount = 0;
    debouncePos(tensCand, tensPos, tensPending, tensPendingCount, KNOB_CONFIRM);

    // Ones: 11-pos (labeled 1.02, 2–11), 1MΩ pullup to 3.3V, GPIO 34.
    // Tighter spacing than tens — reduced hysteresis (15) keeps all 11
    // positions reachable in both grounded and ungrounded modes.
    static const int ONES_THRESH[11] =
        { 1950, 1846, 1736, 1624, 1478, 1330, 1161, 971, 743, 408, 93 };
    static const int ONES_HYST = 15;
    int onesCand = decodeBinHyst(onesRaw, onesPos, ONES_THRESH, 11, ONES_HYST);
    static int onesPending = -1, onesPendingCount = 0;
    debouncePos(onesCand, onesPos, onesPending, onesPendingCount, KNOB_CONFIRM);

    // Hundreds: 5-pos rotary switch, 10kΩ pullup to 3.3V, GPIO 35
    // Measured ADC readings (±20 variance):
    //   Pos 1 (8.8kΩ): 1780
    //   Pos 2 (6.6kΩ): 1480
    //   Pos 3 (4.4kΩ): 1100
    //   Pos 4 (2.2kΩ):  570
    //   Pos 5 (short):    0
    static const int HUNDREDS_THRESH[4] = { 1630, 1290, 835, 285 };
    int hundredsCand =
        decodeBinHyst(hundredsRaw, hundredsPos, HUNDREDS_THRESH, 4, KNOB_HYST);
    static int hundredsPending = -1, hundredsPendingCount = 0;
    debouncePos(hundredsCand, hundredsPos, hundredsPending,
                hundredsPendingCount, KNOB_CONFIRM);

    if (adsOk) {
        adsTensRaw = ads.readADC_SingleEnded(0);
        adsTensV = ads.computeVolts(adsTensRaw);
        lastAdsTensRaw = adsTensRaw;

        ads.setGain(GAIN_TWO);
        delay(5);
        int32_t onesSum = 0;
        for (int i = 0; i < 4; i++) onesSum += ads.readADC_SingleEnded(1);
        adsOnesRaw = onesSum / 4;
        adsOnesV = adsOnesRaw * ADS_LSB_GAIN2;
        lastAdsOnesRaw = adsOnesRaw;
        ads.setGain(GAIN_ONE);

        lastAdsVccRef = ads.readADC_SingleEnded(2);

        // Ratiometric correction: scale ADS readings to match calibration-time Vcc.
        float vccScale = (lastAdsVccRef > 1000)
            ? adsVccRefBaseline / (float)lastAdsVccRef : 1.0f;
        int16_t adsTensCorrected = (int16_t)(adsTensRaw * vccScale);
        int16_t adsOnesCorrected = (int16_t)(adsOnesRaw * vccScale);

        float onesR = onesRawToR(adsOnesCorrected);
        int decInnerCand = (int)(onesR + 0.5);
        if (decInnerCand > 9) decInnerCand = 9;
        if (decInnerCand < 0) decInnerCand = 0;
        static int decInnerPending = -1, decInnerPendingCount = 0;
        debouncePos(decInnerCand, decInner, decInnerPending, decInnerPendingCount, 2);

        int newH = 0;
        for (int i = 10; i >= 0; i--) {
            if (adsTensCorrected >= hundredsThresh[i]) { newH = i + 1; break; }
        }
        int bracketLow = hundredsLUT[newH];
        int bracketHigh = hundredsLUT[newH + 1];
        float countsPerTen = (bracketHigh - bracketLow) / 10.0;
        int newT = (int)((adsTensCorrected - bracketLow) / countsPerTen + 0.5);
        if (newT > 9) newT = 9;
        if (newT < 0) newT = 0;

        int newVal = newH * 100 + newT * 10 + decInner;
        if (newVal != decValue) {
            if (newVal == decCandValue) {
                decConfirm++;
                if (decConfirm >= 2) {
                    decOuter = newH;
                    decMid = newT;
                    decValue = newVal;
                    decCandValue = -1;
                    decConfirm = 0;
                }
            } else {
                decCandValue = newVal;
                decConfirm = 1;
            }
        } else {
            decCandValue = -1;
            decConfirm = 0;
        }
    }

    // Meters track knob position: uA ← brightness (tens), V ← speed
    // (hundreds). Needle maps the knob across its full travel, 0 at the
    // minimum detent → full scale at the top, so the lowest position reads
    // a true zero. A knob move re-arms auto (command-override clears).
    static int prevTensForMeter = -1, prevHundredsForMeter = -1;
    if (tensPos != prevTensForMeter)         { uaAuto = true; prevTensForMeter = tensPos; }
    if (hundredsPos != prevHundredsForMeter) { vAuto  = true; prevHundredsForMeter = hundredsPos; }

    if (uaAuto) {
        // µA meter: tens 0–9 → 0,10,20…90 on 500µA face.
        int t = tensPos > 9 ? 9 : tensPos;
        uaDac = (int)((t * 10.0f / 500.0f) * METER_UA_MAX + 0.5f);
    }
    if (vAuto) {
        // V meter: hundreds 0–4 → 0,100,200,300,400 on 500V face.
        int h = hundredsPos > 4 ? 4 : hundredsPos;
        vDac = (int)((h * 100.0f / 500.0f) * METER_V_MAX + 0.5f);
    }

    int uaVal = uaDac > METER_UA_MAX ? METER_UA_MAX : uaDac;
    int vVal  = vDac  > METER_V_MAX  ? METER_V_MAX  : vDac;

    // digitalWrite gives true 0V; dacWrite(pin, 0) leaks ~80mV
    if (uaVal == 0) { pinMode(METER_UA_PIN, OUTPUT); digitalWrite(METER_UA_PIN, LOW); }
    else dacWrite(METER_UA_PIN, uaVal);
    if (vVal == 0) { pinMode(METER_V_PIN, OUTPUT); digitalWrite(METER_V_PIN, LOW); }
    else dacWrite(METER_V_PIN, vVal);

    // Change detection
    stateChanged = (dcOn != prevDcOn || acOn != prevAcOn ||
                    hundredsPos != prevHundredsPos || tensPos != prevTensPos ||
                    onesPos != prevOnesPos || decValue != prevDecValue ||
                    uaDac != prevUaDac || vDac != prevVDac ||
                    METER_UA_MAX != prevUaMax || METER_V_MAX != prevVMax);

    if (stateChanged || millis() - lastBroadcast >= HEARTBEAT_MS) {
        broadcastState();
        lastBroadcast = millis();
        prevDcOn = dcOn; prevAcOn = acOn;
        prevHundredsPos = hundredsPos; prevTensPos = tensPos;
        prevOnesPos = onesPos; prevDecValue = decValue;
        prevUaDac = uaDac; prevVDac = vDac;
        prevUaMax = METER_UA_MAX; prevVMax = METER_V_MAX;
    }

    // Serial debug output
    Serial.printf("{\"dc\":%s,\"ac\":%s,\"hundreds\":%d,\"tens\":%d,\"ones\":%d,"
                  "\"decade\":%d,\"decouter\":%d,\"decmid\":%d,\"decinner\":%d,"
                  "\"ua\":{\"dac\":%d,\"max\":%d},\"v\":{\"dac\":%d,\"max\":%d},"
                  "\"raw\":{\"hundreds\":%d,\"tens\":%d,\"ones\":%d},\"seq\":%u}\n",
                  dcOn ? "true" : "false", acOn ? "true" : "false",
                  hundredsPos, tensPos, onesPos,
                  decValue, decOuter, decMid, decInner,
                  uaDac, METER_UA_MAX, vDac, METER_V_MAX,
                  hundredsRaw, tensRaw, onesRaw, espnowSeq);

    delay(100);
}
