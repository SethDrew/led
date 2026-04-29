/*
 * I2C_PROBE — detect MPU-6050 (or any I2C device) on a freshly-connected board.
 *
 * Scans the I2C bus, reports every device that ACKs, and explicitly checks
 * 0x68 / 0x69 (MPU-6050 with AD0 low/high) and reads WHO_AM_I (reg 0x75)
 * to confirm it's an MPU-6050 (returns 0x68 or 0x70/0x71 on common clones).
 *
 * Canonical bulb-sender wiring on ESP32-C3:
 *   SDA = GPIO 8
 *   SCL = GPIO 7
 *   AD0 = GPIO 20 (driven LOW at boot to force MPU addr 0x68)
 */

#include <Arduino.h>
#include <Wire.h>

#ifndef PROBE_SDA
#define PROBE_SDA 8
#endif
#ifndef PROBE_SCL
#define PROBE_SCL 7
#endif
#ifndef PROBE_AD0
#define PROBE_AD0 20
#endif

static void scanBus() {
    Serial.println("[probe] scanning 0x03..0x77");
    int found = 0;
    for (uint8_t addr = 0x03; addr <= 0x77; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            Serial.printf("[probe]   ack 0x%02X\n", addr);
            found++;
        }
    }
    Serial.printf("[probe] %d device(s) total\n", found);
}

static void checkMpu(uint8_t addr) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() != 0) {
        Serial.printf("[probe] no device at 0x%02X\n", addr);
        return;
    }
    // WHO_AM_I @ 0x75
    Wire.beginTransmission(addr);
    Wire.write(0x75);
    Wire.endTransmission(false);
    Wire.requestFrom(addr, (uint8_t)1);
    if (Wire.available()) {
        uint8_t whoami = Wire.read();
        const char* tag = "unknown";
        if (whoami == 0x68) tag = "MPU-6050";
        else if (whoami == 0x70 || whoami == 0x71) tag = "MPU-6050 clone / MPU-9250";
        else if (whoami == 0x12 || whoami == 0x14) tag = "MPU-6500";
        Serial.printf("[probe] device at 0x%02X: WHO_AM_I=0x%02X (%s)\n",
                      addr, whoami, tag);
    } else {
        Serial.printf("[probe] device at 0x%02X but WHO_AM_I read failed\n", addr);
    }
}

void setup() {
    Serial.begin(460800);
    delay(800);
    Serial.println();
    Serial.println("=== I2C probe ===");
    Serial.printf("SDA=%d SCL=%d AD0=%d (driven LOW)\n",
                  PROBE_SDA, PROBE_SCL, PROBE_AD0);

    pinMode(PROBE_AD0, OUTPUT);
    digitalWrite(PROBE_AD0, LOW);   // force MPU addr to 0x68

    Wire.begin(PROBE_SDA, PROBE_SCL);
    Wire.setClock(100000);  // slow for reliability on probe
    delay(50);

    scanBus();
    Serial.println("--- MPU check ---");
    checkMpu(0x68);
    checkMpu(0x69);
    Serial.println("=== done — looping in 5s ===");
}

void loop() {
    delay(5000);
    scanBus();
    checkMpu(0x68);
    checkMpu(0x69);
}
