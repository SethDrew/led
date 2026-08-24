#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <driver/i2s.h>

#ifndef PROBE_SDA
#define PROBE_SDA 3
#endif
#ifndef PROBE_SCL
#define PROBE_SCL 4
#endif
#ifndef PROBE_CS
#define PROBE_CS 0
#endif
#ifndef PROBE_ADD
#define PROBE_ADD 1
#endif
// INT is not wired on the current bench setup; set >=0 when it is.
#ifndef PROBE_INT
#define PROBE_INT -1
#endif
#ifndef PROBE_I2S_WS
#define PROBE_I2S_WS 21
#endif
#ifndef PROBE_I2S_SCK
#define PROBE_I2S_SCK 20
#endif
// GPIO9 is the BOOT strapping pin; the INMP441 tri-states SD unclocked and the
// board pullup holds it high, but a no-boot with mic attached points here.
#ifndef PROBE_I2S_SD
#define PROBE_I2S_SD 9
#endif
#ifndef PROBE_MIC
#define PROBE_MIC 1
#endif
// Wire the module's RST pad here and wedged-sensor recovery becomes automatic
// (the BNO08x can wedge such that only a hard reset revives it).
#ifndef PROBE_RST
#define PROBE_RST -1
#endif
// The 90-pair init sweep takes ~30 s; skip it when using the meter assist.
#ifndef PROBE_SWEEP
#define PROBE_SWEEP 0
#endif
// Park SCL at a static level so a DC meter reading is unambiguous.
#ifndef PROBE_METER_HOLD
#define PROBE_METER_HOLD 0
#endif
// Live per-wire continuity readout: reseat a wire and watch it change.
#ifndef PROBE_MONITOR
#define PROBE_MONITOR 0
#endif

// The driver's own hard-reset-in-begin leaves this module handshaking but
// never streaming; RST is pulsed manually in tryInit as a last resort instead.
static Adafruit_BNO08x bno(-1);
static sh2_SensorValue_t ev;

static bool     up = false;
static uint8_t  addr = 0;
static uint32_t foundHz = 0;
static uint32_t lastScan = 0;
static uint32_t lastPrint = 0;
static bool     deepDone = false;

static float linPk = 0;
static float qi, qj, qk, qr = 1.0f;
static float rvi, rvj, rvk, rvr = 1.0f, rvacc = -1.0f;
static float lax, lay, laz;
static float gx, gy, gz;
static float ax, ay, az;
static float grvx, grvy, grvz;
static float mgx, mgy, mgz;
static uint32_t steps = 0;
static uint8_t stability = 0;
static uint8_t activity = 0, actConf = 0;
static uint32_t taps = 0, shakes = 0;
static uint32_t events = 0;

static bool     micUp = false;
static float    micRms = -120.0f, micPk = -120.0f;
static uint32_t micSamples = 0;

static volatile uint32_t irqs = 0;
static void IRAM_ATTR intIsr() { irqs++; }

static const char* STABILITY[] = {"unknown", "on-table", "stationary",
                                  "stable", "in-motion", "reserved"};

static const uint32_t SPEEDS[] = {100000, 50000, 20000, 10000};

static uint32_t riseUs(uint8_t pin);

static void csIdleHigh() {
    pinMode(PROBE_CS, OUTPUT);
    digitalWrite(PROBE_CS, HIGH);
}

static void micInit() {
    if (!PROBE_MIC) return;
    i2s_config_t cfg = {};
    cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
    cfg.sample_rate = 16000;
    cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
    cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    cfg.dma_buf_count = 4;
    cfg.dma_buf_len = 256;

    i2s_pin_config_t pins = {};
    pins.mck_io_num = I2S_PIN_NO_CHANGE;
    pins.bck_io_num = PROBE_I2S_SCK;
    pins.ws_io_num = PROBE_I2S_WS;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = PROBE_I2S_SD;

    micUp = i2s_driver_install(I2S_NUM_0, &cfg, 0, nullptr) == ESP_OK &&
            i2s_set_pin(I2S_NUM_0, &pins) == ESP_OK;
    Serial.printf("[mic] INMP441 i2s WS=%d SCK=%d SD=%d ... %s\n",
                  PROBE_I2S_WS, PROBE_I2S_SCK, PROBE_I2S_SD,
                  micUp ? "ok" : "FAILED");
}

static void micRead() {
    if (!micUp) return;
    static int32_t buf[256];
    float sum = 0, pk = 0;
    int n = 0;
    size_t got = 0;
    while (i2s_read(I2S_NUM_0, buf, sizeof(buf), &got, 0) == ESP_OK && got) {
        int cnt = got / 4;
        for (int i = 0; i < cnt; i++) {
            float v = (float)buf[i] / 2147483648.0f;
            sum += v * v;
            float a = fabsf(v);
            if (a > pk) pk = a;
        }
        n += cnt;
        if (got < sizeof(buf)) break;
    }
    if (n) {
        micSamples += n;
        micRms = 20.0f * log10f(sqrtf(sum / n) + 1e-12f);
        micPk = 20.0f * log10f(pk + 1e-12f);
    }
}

static uint8_t scanAt(uint32_t hz) {
    Wire.end();
    delay(2);
    Wire.begin(PROBE_SDA, PROBE_SCL);
    Wire.setClock(hz);
    Wire.setTimeOut(1000);
    delay(5);

    uint8_t found = 0;
    Serial.printf("[scan %6lu Hz]", (unsigned long)hz);
    for (uint8_t a = 0x08; a < 0x78; a++) {
        Wire.beginTransmission(a);
        if (Wire.endTransmission() == 0) {
            Serial.printf(" 0x%02X", a);
            found++;
            if (a == 0x4A || a == 0x4B) { addr = a; foundHz = hz; }
        }
    }
    if (!found) Serial.print(" -");
    Serial.println();
    return found;
}

static uint8_t scanAllSpeeds() {
    csIdleHigh();
    for (uint32_t hz : SPEEDS)
        if (scanAt(hz)) return 1;
    return 0;
}

static uint8_t scanPair(uint8_t sda, uint8_t scl, uint32_t hz) {
    Wire.end();
    delay(2);
    Wire.begin(sda, scl);
    Wire.setClock(hz);
    Wire.setTimeOut(1000);
    delay(5);
    for (uint8_t a = 0x08; a < 0x78; a++) {
        Wire.beginTransmission(a);
        if (Wire.endTransmission() == 0) {
            Serial.printf("[FOUND] 0x%02X on SDA=%d SCL=%d @ %lu Hz\n",
                          a, sda, scl, (unsigned long)hz);
            if (a == 0x4A || a == 0x4B) { addr = a; foundHz = hz; }
            return 1;
        }
    }
    return 0;
}

// Run before anything reconfigures the pins. 11-17 are SPI flash, 18/19 USB.
static uint8_t sweepPins() {
    const uint8_t P[] = {0, 1, 2, 3, 4, 5, 6, 7, 10, 20, 21};
    csIdleHigh();
    Serial.println("[sweep] every SDA/SCL pair at 100k and 20k ...");
    for (uint32_t hz : (const uint32_t[]){100000, 20000})
        for (uint8_t sda : P)
            for (uint8_t scl : P) {
                if (sda == scl || sda == PROBE_CS || scl == PROBE_CS) continue;
                if (scanPair(sda, scl, hz)) return 1;
            }
    Serial.println("[sweep] no device on any pin pair");
    return 0;
}

static void riseAll() {
    const uint8_t P[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21};
    Serial.println("[rise] pullup strength, every exposed pin:");
    for (uint8_t p : P) {
        uint32_t t = riseUs(p);
        Serial.printf("  GPIO%-2d  %s\n", p,
                      t == 0    ? "none"
                      : t <= 3  ? "STRONG (2-10k)"
                      : t <= 40 ? "weak (~45k)"
                                : "very weak");
    }
}

// Drive low, release to a pull-free input, and time the recovery. A few-k
// pullup snaps back in microseconds; leakage alone never gets there. This
// measures pullup STRENGTH, where a static level read only guesses at it.
static uint32_t riseUs(uint8_t pin) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);
    delayMicroseconds(200);
    pinMode(pin, INPUT);
    uint32_t t0 = micros();
    while (!digitalRead(pin))
        if (micros() - t0 > 50000) return UINT32_MAX;
    return micros() - t0;
}

static bool hasPullup(uint8_t pin) {
    pinMode(pin, OUTPUT_OPEN_DRAIN);
    digitalWrite(pin, LOW);
    delayMicroseconds(500);
    digitalWrite(pin, HIGH);
    delayMicroseconds(1000);
    bool rel = digitalRead(pin);

    pinMode(pin, INPUT_PULLDOWN);
    delayMicroseconds(2000);
    bool pd = digitalRead(pin);

    pinMode(pin, INPUT);
    return rel && pd;
}

// arduino-esp32 OUTPUT is (INPUT|OUTPUT), so the input buffer stays live and a
// pin can be read back while it is driving.
static void selfCheck(uint8_t pin) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);
    delayMicroseconds(500);
    int ppLo = digitalRead(pin);

    digitalWrite(pin, HIGH);
    delayMicroseconds(500);
    int ppHi = digitalRead(pin);

    pinMode(pin, OUTPUT_OPEN_DRAIN);
    digitalWrite(pin, LOW);
    delayMicroseconds(500);
    digitalWrite(pin, HIGH);
    delayMicroseconds(1000);
    int rel = digitalRead(pin);

    pinMode(pin, INPUT);
    delayMicroseconds(500);
    int flt = digitalRead(pin);

    pinMode(pin, INPUT_PULLUP);
    delayMicroseconds(500);
    int pu = digitalRead(pin);

    pinMode(pin, INPUT_PULLDOWN);
    delayMicroseconds(2000);
    int pd = digitalRead(pin);

    pinMode(pin, INPUT);
    uint32_t t = riseUs(pin);
    pinMode(pin, INPUT);

    char rise[12];
    if (t == UINT32_MAX) strcpy(rise, "  never");
    else                 snprintf(rise, sizeof(rise), "%7lu", (unsigned long)t);

    Serial.printf("  %-2d   %d    %d    %d   %d   %d   %d  %s%s\n",
                  pin, ppLo, ppHi, rel, flt, pu, pd, rise,
                  ppLo ? "   <== CANNOT DRIVE LOW" : "");
}

static void reportRise(const char* name, uint8_t pin) {
    uint32_t t = riseUs(pin);
    const char* verdict;
    if (t == UINT32_MAX) verdict = "NO pullup (never rose)";
    else if (t <= 3)     verdict = "strong pullup, ~2-10k";
    else if (t <= 40)    verdict = "weak pullup (internal-ish, ~45k)";
    else                 verdict = "very weak / leakage only";
    Serial.printf("[rise] %-9s GPIO%-2d  %6lu us  %s\n",
                  name, pin, (unsigned long)t, verdict);
}

static void deepDiag() {
    Serial.println("--- deep diagnostic ---");
    reportRise("SDA", PROBE_SDA);
    reportRise("SCL", PROBE_SCL);
    reportRise("CS", PROBE_CS);
    reportRise("ADD", PROBE_ADD);
    if (PROBE_INT >= 0) reportRise("INT", PROBE_INT);

    // If PS1:PS0 latched 11 the part free-runs 19-byte 0xAA 0xAA UART frames
    // at 115200 instead of answering I2C at all.
    const int pins[2] = {PROBE_SDA, PROBE_SCL};
    for (int i = 0; i < 2; i++) {
        Serial1.begin(115200, SERIAL_8N1, pins[i], -1);
        delay(60);
        uint8_t prev = 0;
        int bytes = 0, hdr = 0;
        uint32_t t0 = millis();
        while (millis() - t0 < 250)
            while (Serial1.available()) {
                uint8_t b = Serial1.read();
                bytes++;
                if (prev == 0xAA && b == 0xAA) hdr++;
                prev = b;
            }
        Serial1.end();
        Serial.printf("[rvc] GPIO%-2d: %d bytes, %d AA-AA%s\n", pins[i], bytes,
                      hdr, hdr > 5 ? "  <<< UART-RVC MODE" : "");
    }
    pinMode(PROBE_SDA, INPUT);
    pinMode(PROBE_SCL, INPUT);
    Serial.println("-----------------------");
}

static void enableReports() {
    struct { sh2_SensorId_t id; uint32_t us; const char* name; } reports[] = {
        {SH2_GAME_ROTATION_VECTOR,      20000, "game-rotation"},
        {SH2_ROTATION_VECTOR,           20000, "rotation-vector"},
        {SH2_LINEAR_ACCELERATION,       20000, "linear-accel"},
        {SH2_ACCELEROMETER,             20000, "accelerometer"},
        {SH2_GRAVITY,                   40000, "gravity"},
        {SH2_GYROSCOPE_CALIBRATED,      20000, "gyro"},
        {SH2_MAGNETIC_FIELD_CALIBRATED, 40000, "magnetometer"},
        {SH2_TAP_DETECTOR,                  0, "tap"},
        {SH2_SHAKE_DETECTOR,                0, "shake"},
        {SH2_STEP_COUNTER,             200000, "step-counter"},
        {SH2_STABILITY_CLASSIFIER,     200000, "stability"},
    };
    for (auto& r : reports)
        Serial.printf("[report] %-15s %s\n", r.name,
                      bno.enableReport(r.id, r.us) ? "ok" : "REJECTED");

    // The activity classifier needs its enabled-activities bitmask in
    // sensorSpecific; the generic enableReport leaves it 0 = nothing enabled.
    sh2_SensorConfig_t cfg = {};
    cfg.reportInterval_us = 500000;
    cfg.sensorSpecific = 0x1FF;
    Serial.printf("[report] %-15s %s\n", "activity",
                  sh2_setSensorConfig(SH2_PERSONAL_ACTIVITY_CLASSIFIER, &cfg)
                      == SH2_OK ? "ok" : "REJECTED");
}

static void rampClock() {
    const uint32_t speeds[] = {400000, 200000, 100000, 50000, 20000, 10000};
    for (uint32_t hz : speeds) {
        Wire.setClock(hz);
        delay(50);
        uint32_t got = 0, t0 = millis();
        while (millis() - t0 < 800)
            if (bno.getSensorEvent(&ev)) got++;
        Serial.printf("[clock] %6lu Hz: %3lu events in 0.8 s%s\n",
                      (unsigned long)hz, (unsigned long)got,
                      got > 20 ? "  <== settled here" : "");
        if (got > 20) { foundHz = hz; return; }
    }
    Wire.setClock(10000);
    foundHz = 10000;
}

static void tryInit() {
    if (!addr) return;
    Serial.printf("[init] BNO08x at 0x%02X (%lu Hz) ... ", addr,
                  (unsigned long)foundHz);
    bool ok = bno.begin_I2C(addr, &Wire, 0);
    // Boot-time clock stretching can pass an address scan yet kill the SHTP
    // handshake; a 10 kHz handshake always lands, and rampClock restores speed.
    if (!ok && foundHz > 10000) {
        Serial.print("retry at 10 kHz ... ");
        Wire.setClock(10000);
        foundHz = 10000;
        ok = bno.begin_I2C(addr, &Wire, 0);
    }
    // A wedged BNO08x ACKs but won't handshake; only a hard reset clears it.
    // Pulse RST then release to input — the module pullup owns the line.
    if (!ok && PROBE_RST >= 0) {
        Serial.print("RST pulse ... ");
        pinMode(PROBE_RST, OUTPUT);
        digitalWrite(PROBE_RST, LOW);
        delay(20);
        pinMode(PROBE_RST, INPUT);
        delay(600);
        ok = bno.begin_I2C(addr, &Wire, 0);
    }
    if (!ok) { Serial.println("failed"); return; }
    Serial.println("ok");
    for (int i = 0; i < bno.prodIds.numEntries; i++) {
        sh2_ProductId_t& p = bno.prodIds.entry[i];
        Serial.printf("[id] part %lu  sw %u.%u.%lu  build %lu\n",
                      (unsigned long)p.swPartNumber, p.swVersionMajor,
                      p.swVersionMinor, (unsigned long)p.swVersionPatch,
                      (unsigned long)p.swBuildNumber);
    }
    enableReports();
    rampClock();
    up = true;
}

void setup() {
    Serial.begin(460800);
    // Drop output when no host drains CDC — a full TX buffer otherwise
    // blocks printf forever, freezing the probe when the dashboard closes.
    Serial.setTxTimeoutMs(0);
    delay(600);
    Serial.println("\n=== BNO08x probe ===");
    Serial.printf("SDA=%d SCL=%d CS=%d ADD=%d INT=%d\n",
                  PROBE_SDA, PROBE_SCL, PROBE_CS, PROBE_ADD, PROBE_INT);

    if (PROBE_MONITOR) {
        Serial.println("\n=== live wire monitor ===");
        Serial.println("Reseat one wire at a time and watch its column flip.");
        Serial.println("GPIO5/6 are unwired controls: they must stay 'open'.");
        Serial.println("Nothing here talks to the BNO08x; it only senses the");
        Serial.println("module's own pullups appearing on each line.\n");
        return;
    }

    if (PROBE_METER_HOLD) {
        Serial.println("\n=== pin self-check (C3 reading its own pins) ===");
        Serial.println("gpio  ppLo ppHi rel flt  pu  pd   riseUs");
        const uint8_t pins[] = {PROBE_CS, PROBE_ADD, PROBE_SDA, PROBE_SCL, 4, 5, 6};
        for (uint8_t p : pins) selfCheck(p);
        Serial.println("ppLo=1 -> pin cannot be pulled low: short to 3V3");
        Serial.println("rel=1  -> external pullup present   pd=1 -> pullup beats 45k");
        Serial.println("GPIO5/6 are unwired controls; GPIO2 is a strapping pin");

        pinMode(PROBE_SCL, OUTPUT);
        digitalWrite(PROBE_SCL, LOW);
        Serial.printf("\n*** GPIO%d NOW HELD LOW ***  measure vs a C3 GND pin\n",
                      PROBE_SCL);
        Serial.printf("  C3 GPIO%d pin      -> 0 V expected\n", PROBE_SCL);
        Serial.println("  module SCL copper -> 0 V = connected, 3.3 V = open");
        return;
    }

    // BNO08x picks its host interface from PS1:PS0 (00 = I2C) and needs nCS
    // held HIGH in I2C mode. RST must not be left held low.
    csIdleHigh();
    pinMode(PROBE_ADD, INPUT);
    if (PROBE_INT >= 0) {
        pinMode(PROBE_INT, INPUT_PULLUP);
        attachInterrupt(digitalPinToInterrupt(PROBE_INT), intIsr, FALLING);
    }

    // Clean scan before any diagnostic touches the pins — the diagnostics
    // themselves reconfigure SDA/SCL and can leave the bus unusable.
    riseAll();
    micInit();
    if (!PROBE_SWEEP) return;

    // A bare address ACK can be a false positive on a marginal bus; only a
    // completed SHTP product-ID exchange proves the pin pair is really right.
    const uint8_t P[] = {0, 2, 3, 4, 5, 6, 7, 10, 20, 21};
    for (uint8_t sda : P)
        for (uint8_t scl : P) {
            if (sda == scl) continue;
            Wire.end();
            delay(2);
            Wire.begin(sda, scl);
            Wire.setClock(100000);
            Wire.setTimeOut(1000);
            delay(5);
            for (uint8_t a : (const uint8_t[]){0x4A, 0x4B}) {
                Wire.beginTransmission(a);
                if (Wire.endTransmission() != 0) continue;
                Serial.printf("[ack] 0x%02X on SDA=%d SCL=%d — trying init ... ",
                              a, sda, scl);
                if (bno.begin_I2C(a, &Wire, 0)) {
                    Serial.println("OK");
                    addr = a; foundHz = 100000;
                    Serial.printf("[pins] CONFIRMED SDA=%d SCL=%d addr=0x%02X\n",
                                  sda, scl, a);
                    for (int i = 0; i < bno.prodIds.numEntries; i++) {
                        sh2_ProductId_t& p = bno.prodIds.entry[i];
                        Serial.printf("[id] part %lu  sw %u.%u.%lu  build %lu\n",
                                      (unsigned long)p.swPartNumber,
                                      p.swVersionMajor, p.swVersionMinor,
                                      (unsigned long)p.swVersionPatch,
                                      (unsigned long)p.swBuildNumber);
                    }
                    enableReports();
                    up = true;
                    return;
                }
                Serial.println("false positive");
            }
        }
    Serial.println("[sweep] no pin pair completed init");
}

void loop() {
    if (PROBE_MONITOR) {
        struct Line { const char* name; uint8_t pin; };
        static const Line lines[] = {
            {"CS",   PROBE_CS},  {"ADD",  PROBE_ADD},
            {"SDA",  PROBE_SDA}, {"SCL",  PROBE_SCL},
            {"ctl5", 5},         {"ctl6", 6},
        };
        static bool prev[6];
        static bool primed = false;

        bool changed = false;
        char row[128] = "";
        for (size_t i = 0; i < 6; i++) {
            bool c = hasPullup(lines[i].pin);
            if (primed && c != prev[i]) changed = true;
            prev[i] = c;
            char cell[24];
            snprintf(cell, sizeof(cell), "%s:g%d %s | ",
                     lines[i].name, lines[i].pin, c ? "[OK]" : "----");
            strlcat(row, cell, sizeof(row));
        }
        primed = true;
        Serial.printf("%s%s\r\n", row, changed ? "<== CHANGED" : "");
        delay(250);
        return;
    }

    if (PROBE_METER_HOLD) {
        Serial.printf("[hold] GPIO%d driven LOW, reads back %d\n",
                      PROBE_SCL, digitalRead(PROBE_SCL));
        delay(2000);
        return;
    }

    uint32_t now = millis();

    if (!up) {
        if (now - lastScan >= 6000) {
            lastScan = now;
            csIdleHigh();
            if (PROBE_INT >= 0) pinMode(PROBE_INT, INPUT_PULLUP);
            if (scanAllSpeeds()) tryInit();
            else if (!deepDone) { deepDiag(); deepDone = true; }
        }
    }

    if (up && bno.wasReset()) {
        Serial.println("[warn] sensor reset — re-enabling reports");
        enableReports();
    }

    while (up && bno.getSensorEvent(&ev)) {
        events++;
        switch (ev.sensorId) {
            case SH2_GAME_ROTATION_VECTOR:
                qi = ev.un.gameRotationVector.i; qj = ev.un.gameRotationVector.j;
                qk = ev.un.gameRotationVector.k; qr = ev.un.gameRotationVector.real;
                break;
            case SH2_ROTATION_VECTOR:
                rvi = ev.un.rotationVector.i; rvj = ev.un.rotationVector.j;
                rvk = ev.un.rotationVector.k; rvr = ev.un.rotationVector.real;
                rvacc = ev.un.rotationVector.accuracy;
                break;
            case SH2_LINEAR_ACCELERATION:
                lax = ev.un.linearAcceleration.x; lay = ev.un.linearAcceleration.y;
                laz = ev.un.linearAcceleration.z;
                linPk = fmaxf(linPk, fmaxf(fabsf(lax), fmaxf(fabsf(lay), fabsf(laz))));
                break;
            case SH2_ACCELEROMETER:
                ax = ev.un.accelerometer.x; ay = ev.un.accelerometer.y;
                az = ev.un.accelerometer.z;
                break;
            case SH2_GRAVITY:
                grvx = ev.un.gravity.x; grvy = ev.un.gravity.y;
                grvz = ev.un.gravity.z;
                break;
            case SH2_GYROSCOPE_CALIBRATED:
                gx = ev.un.gyroscope.x; gy = ev.un.gyroscope.y; gz = ev.un.gyroscope.z;
                break;
            case SH2_MAGNETIC_FIELD_CALIBRATED:
                mgx = ev.un.magneticField.x; mgy = ev.un.magneticField.y;
                mgz = ev.un.magneticField.z;
                break;
            case SH2_TAP_DETECTOR:
                taps++;
                Serial.printf("{\"e\":\"tap\",\"f\":%u,\"n\":%lu}\n",
                              ev.un.tapDetector.flags, (unsigned long)taps);
                break;
            case SH2_SHAKE_DETECTOR:
                shakes++;
                Serial.printf("{\"e\":\"shake\",\"a\":%u,\"n\":%lu}\n",
                              ev.un.shakeDetector.shake, (unsigned long)shakes);
                break;
            case SH2_STEP_COUNTER:
                steps = ev.un.stepCounter.steps;
                break;
            case SH2_STABILITY_CLASSIFIER:
                stability = ev.un.stabilityClassifier.classification;
                break;
            case SH2_PERSONAL_ACTIVITY_CLASSIFIER:
                activity = ev.un.personalActivityClassifier.mostLikelyState;
                actConf = ev.un.personalActivityClassifier
                              .confidence[activity < 10 ? activity : 0];
                break;
        }
    }

    if (now - lastPrint >= 40) {
        lastPrint = now;
        micRead();
        Serial.printf(
            "{\"q\":[%.4f,%.4f,%.4f,%.4f],\"rq\":[%.4f,%.4f,%.4f,%.4f],"
            "\"ra\":%.3f,\"lin\":[%.2f,%.2f,%.2f],\"acc\":[%.2f,%.2f,%.2f],"
            "\"grv\":[%.2f,%.2f,%.2f],\"gyr\":[%.3f,%.3f,%.3f],"
            "\"mag\":[%.1f,%.1f,%.1f],\"stp\":%lu,\"stb\":%u,\"act\":%u,"
            "\"ac\":%u,\"tap\":%lu,\"shk\":%lu,\"ev\":%lu,\"hz\":%lu,"
            "\"mic\":[%.1f,%.1f],\"mn\":%lu,\"irq\":%lu,\"lpk\":%.2f}\n",
            qi, qj, qk, qr, rvi, rvj, rvk, rvr, rvacc,
            lax, lay, laz, ax, ay, az, grvx, grvy, grvz, gx, gy, gz,
            mgx, mgy, mgz, (unsigned long)steps, stability, activity, actConf,
            (unsigned long)taps, (unsigned long)shakes, (unsigned long)events,
            (unsigned long)foundHz, micRms, micPk, (unsigned long)micSamples,
            (unsigned long)irqs, linPk);
        linPk = 0;
    }
}
