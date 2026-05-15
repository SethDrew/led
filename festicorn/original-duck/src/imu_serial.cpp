/*
 * IMU_SERIAL — duck variant for laptop-driven record/playback (topology A).
 *
 * Strips ESP-NOW and I2S. Reads only the MPU-6050 over I2C and emits
 * a 15-byte SensorPacket (matching the original wire shape) framed on
 * USB serial at 25 Hz. The laptop fills in the audio fields from its own
 * mic before forwarding the merged packet to the bridge.
 *
 * Frame format (matches espnow-bridge bridge.cpp):
 *   [0xA5][0x5A][LEN:1][PAYLOAD: LEN bytes][XOR8: LEN ^ payload bytes]
 *
 * Wiring (unchanged):
 *   GPIO 21 → SCL (MPU-6050)
 *   GPIO 20 → SDA (MPU-6050)
 */

#include <Arduino.h>
#include <Wire.h>

#ifndef SDA_PIN
#define SDA_PIN  20
#endif
#ifndef SCL_PIN
#define SCL_PIN  21
#endif
#define MPU_ADDR 0x68
#define SENSOR_MS 40   // 25 Hz

static const uint8_t FRAME_M0 = 0xA5;
static const uint8_t FRAME_M1 = 0x5A;

struct __attribute__((packed)) SensorPacket {
    int16_t  ax, ay, az;
    int16_t  gx, gy, gz;
    uint16_t rawRms;       // always 0 in this firmware — laptop fills in
    uint8_t  micEnabled;   // always 0 here
};

static int16_t imuAx, imuAy, imuAz, imuGx, imuGy, imuGz;
static uint32_t lastSensorMs = 0;

static void readIMU() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 14);
    imuAx = (Wire.read() << 8) | Wire.read();
    imuAy = (Wire.read() << 8) | Wire.read();
    imuAz = (Wire.read() << 8) | Wire.read();
    Wire.read(); Wire.read();  // skip temp
    imuGx = (Wire.read() << 8) | Wire.read();
    imuGy = (Wire.read() << 8) | Wire.read();
    imuGz = (Wire.read() << 8) | Wire.read();
}

static void writeFrame(const uint8_t* data, uint8_t len) {
    uint8_t hdr[3] = { FRAME_M0, FRAME_M1, len };
    Serial.write(hdr, 3);
    Serial.write(data, len);
    uint8_t x = len;
    for (int i = 0; i < len; i++) x ^= data[i];
    Serial.write(&x, 1);
}

void setup() {
    Serial.begin(460800);
    delay(300);

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    // Robust MPU wake (same as original sender — handles boards left in
    // CYCLE/WOM mode by a previous firmware).
    for (int attempt = 0; attempt < 40; attempt++) {
        Wire.beginTransmission(MPU_ADDR);
        Wire.write(0x6B);
        Wire.write(0x00);
        Wire.endTransmission(true);
        delay(50);
        Wire.beginTransmission(MPU_ADDR);
        Wire.write(0x75);
        if (Wire.endTransmission(false) == 0) {
            Wire.requestFrom(MPU_ADDR, (uint8_t)1);
            if (Wire.available()) {
                uint8_t v = Wire.read();
                if (v == 0x68 || v == 0x71) break;
            }
        }
    }
    Wire.beginTransmission(MPU_ADDR); Wire.write(0x6C); Wire.write(0x00); Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR); Wire.write(0x1A); Wire.write(0x03); Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR); Wire.write(0x1B); Wire.write(0x00); Wire.endTransmission(true);
    Wire.beginTransmission(MPU_ADDR); Wire.write(0x1C); Wire.write(0x00); Wire.endTransmission(true);

    Serial.printf("[BOOT] role=duck fw=imu_serial rate=25Hz\n");
    lastSensorMs = millis();
}

void loop() {
    // '?' query: re-emit boot banner so a tailer attached mid-run can ID.
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '?') Serial.printf("[BOOT] role=duck fw=imu_serial rate=25Hz\n");
    }

    uint32_t now = millis();
    if (now - lastSensorMs < SENSOR_MS) return;
    lastSensorMs = now;

    readIMU();
    SensorPacket pkt;
    pkt.ax = imuAx; pkt.ay = imuAy; pkt.az = imuAz;
    pkt.gx = imuGx; pkt.gy = imuGy; pkt.gz = imuGz;
    pkt.rawRms = 0;
    pkt.micEnabled = 0;
    writeFrame((const uint8_t*)&pkt, sizeof(pkt));
}
