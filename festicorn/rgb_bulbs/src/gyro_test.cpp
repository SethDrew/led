/*
 * Gyro test — MPU-6050 pin mapping from C3:
 *   GPIO 7  → SCL
 *   GPIO 8  → SDA
 *   GPIO 9  → XDA (auxiliary I2C data, unused)
 *   GPIO 10 → XCL (auxiliary I2C clock, unused)
 *   GPIO 20 → AD0 (address select: LOW=0x68, HIGH=0x69)
 *
 * Continuously re-scans and re-inits so rickety connections still show up
 */

#include <Arduino.h>
#include <Wire.h>

#define SDA_PIN    8
#define SCL_PIN    7
#define AD0_PIN    20
#define MPU_ADDR   0x68

static uint32_t loopCount = 0;
static bool gyroReady = false;

void tryInit() {
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(100000);

    // I2C scan
    int found = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        uint8_t err = Wire.endTransmission(true);
        if (err == 0) {
            Serial.printf("  I2C device at 0x%02X\n", addr);
            found++;
        }
    }

    if (found == 0) {
        Serial.println("  no devices");
        gyroReady = false;
        return;
    }

    // Wake MPU-6050
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B);
    Wire.write(0x00);
    uint8_t err = Wire.endTransmission(true);
    if (err != 0) {
        Serial.printf("  MPU wake failed (err=%d)\n", err);
        gyroReady = false;
        return;
    }

    // WHO_AM_I
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x75);
    Wire.endTransmission(false);
    int n = Wire.requestFrom(MPU_ADDR, 1);
    if (n == 1) {
        uint8_t id = Wire.read();
        Serial.printf("  WHO_AM_I: 0x%02X %s\n", id, id == 0x68 ? "OK" : "UNEXPECTED");
        gyroReady = true;
    } else {
        Serial.println("  WHO_AM_I failed");
        gyroReady = false;
    }
}

void setup() {
    pinMode(AD0_PIN, OUTPUT);
    digitalWrite(AD0_PIN, LOW);

    Serial.begin(460800);
    delay(2000);
    Serial.println("\n=== Gyro test (continuous) ===");
    Serial.printf("SDA=%d SCL=%d AD0=%d\n\n", SDA_PIN, SCL_PIN, AD0_PIN);
}

void loop() {
    loopCount++;

    // Re-scan every 3 seconds if gyro not ready
    if (!gyroReady || loopCount % 30 == 1) {
        Serial.printf("[%lu] scanning...\n", loopCount);
        tryInit();
        if (!gyroReady) {
            delay(1000);
            return;
        }
    }

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    uint8_t err = Wire.endTransmission(false);
    if (err != 0) {
        Serial.printf("[%lu] tx err: %d — rescanning\n", loopCount, err);
        gyroReady = false;
        delay(500);
        return;
    }

    int n = Wire.requestFrom(MPU_ADDR, 14);
    if (n == 14) {
        int16_t ax = (Wire.read() << 8) | Wire.read();
        int16_t ay = (Wire.read() << 8) | Wire.read();
        int16_t az = (Wire.read() << 8) | Wire.read();
        Wire.read(); Wire.read();
        int16_t gx = (Wire.read() << 8) | Wire.read();
        int16_t gy = (Wire.read() << 8) | Wire.read();
        int16_t gz = (Wire.read() << 8) | Wire.read();

        Serial.printf("[%lu] a:%6d %6d %6d  g:%6d %6d %6d\n",
            loopCount, ax, ay, az, gx, gy, gz);
    } else {
        Serial.printf("[%lu] read: %d bytes — rescanning\n", loopCount, n);
        gyroReady = false;
    }

    delay(100);
}
