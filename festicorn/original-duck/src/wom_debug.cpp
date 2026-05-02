// WOM debug: configure MPU-6050 motion-detect, then poll INT_STATUS every
// 100 ms WITHOUT entering light-sleep. Lets us see whether MPU-side motion
// detection actually trips on shake — independent of GPIO routing or
// sleep/wake plumbing.
//
// If shake → "MOTION!" line: MPU detection works, problem is in sleep path.
// If shake → only "quiet" lines: MPU detection itself is misconfigured.
//
// Uses the proposed FIXED register sequence (HPF=0x01 not 0x05, adds
// MOT_DETECT_CTRL=0x15, settle delay before HOLD).

#include <Arduino.h>
#include <Wire.h>

#define SDA_PIN     20
#define SCL_PIN     21
#define MPU_ADDR    0x68
#define WOM_MOT_THR 5    // ~160 mg

static uint8_t mpuRead(uint8_t reg) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return 0xFF;
    Wire.requestFrom(MPU_ADDR, (uint8_t)1);
    return Wire.available() ? Wire.read() : 0xFF;
}

static void mpuWrite(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission(true);
}

static void readAccel(int16_t &ax, int16_t &ay, int16_t &az) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    if (Wire.endTransmission(false) != 0) { ax = ay = az = -1; return; }
    Wire.requestFrom(MPU_ADDR, (uint8_t)6);
    if (Wire.available() < 6) { ax = ay = az = -1; return; }
    ax = (int16_t)((Wire.read() << 8) | Wire.read());
    ay = (int16_t)((Wire.read() << 8) | Wire.read());
    az = (int16_t)((Wire.read() << 8) | Wire.read());
}

void setup() {
    Serial.begin(460800);
    delay(2000);
    Serial.println();
    Serial.println("=== WOM debug (no sleep) ===");

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    // Robust wake from any prior CYCLE state.
    for (int i = 0; i < 40; i++) {
        mpuWrite(0x6B, 0x00);
        delay(50);
        uint8_t who = mpuRead(0x75);
        if (who == 0x68 || who == 0x71) {
            Serial.printf("MPU awake after %d attempts (WHO=0x%02x)\n", i + 1, who);
            break;
        }
    }

    // Configure motion detect (NO CYCLE — stay active so I2C is always live).
    mpuWrite(0x6B, 0x00);  delay(10);   // PWR_MGMT_1: wake
    mpuWrite(0x68, 0x07);  delay(10);   // SIGNAL_PATH_RESET
    mpuWrite(0x1A, 0x00);               // CONFIG: DLPF off
    mpuWrite(0x1C, 0x01);               // ACCEL_CONFIG: FS=±2g, HPF=5Hz  ← FIXED (was 0x05)
    mpuWrite(0x37, 0x20);               // INT_PIN_CFG: LATCH_INT_EN
    mpuWrite(0x38, 0x40);               // INT_ENABLE: MOT_EN
    mpuWrite(0x1F, WOM_MOT_THR);        // MOT_THR
    mpuWrite(0x20, 1);                  // MOT_DUR=1
    mpuWrite(0x69, 0x15);               // MOT_DETECT_CTRL  ← NEW

    delay(200);                         // Let HPF settle around current orientation
    (void)mpuRead(0x3A);                // Clear any latch from configuration

    Serial.printf("Configured: MOT_THR=%d (~%dmg), HPF=5Hz, no CYCLE.\n",
                  WOM_MOT_THR, WOM_MOT_THR * 32);
    Serial.println("Shake the duck. Polling INT_STATUS every 100ms.");
}

void loop() {
    delay(100);

    uint8_t intStatus = mpuRead(0x3A);
    int16_t ax, ay, az;
    readAccel(ax, ay, az);

    static uint32_t pollCount = 0;
    static uint32_t triggers = 0;
    pollCount++;

    if (intStatus & 0x40) {
        triggers++;
        Serial.printf("MOTION!  INT=0x%02x  a=%6d,%6d,%6d  triggers=%lu\n",
                      intStatus, ax, ay, az, (unsigned long)triggers);
    } else if (pollCount % 20 == 0) {
        Serial.printf("quiet    INT=0x%02x  a=%6d,%6d,%6d\n",
                      intStatus, ax, ay, az);
    }
}
