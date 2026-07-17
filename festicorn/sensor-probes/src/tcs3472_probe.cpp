#include <Arduino.h>
#include <Wire.h>

#ifndef PROBE_SDA
#define PROBE_SDA 3
#endif
#ifndef PROBE_SCL
#define PROBE_SCL 2
#endif
#ifndef PROBE_LED
#define PROBE_LED 1
#endif

static const uint8_t ADDR = 0x29;
static const uint8_t CMD = 0x80;

static void wr8(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(ADDR);
    Wire.write(CMD | reg);
    Wire.write(val);
    Wire.endTransmission();
}

static uint16_t rd16(uint8_t reg) {
    Wire.beginTransmission(ADDR);
    Wire.write(CMD | 0x20 | reg);
    Wire.endTransmission(false);
    Wire.requestFrom(ADDR, (uint8_t)2);
    uint16_t lo = Wire.read();
    return lo | (Wire.read() << 8);
}

static const uint8_t GAIN_REGVAL[4] = {0x00, 0x01, 0x02, 0x03};
static uint8_t gainIdx = 1;

static void handleCommand(const String& cmd) {
    if (cmd.startsWith("L ")) {
        int v = constrain(cmd.substring(2).toInt(), 0, 255);
        analogWrite(PROBE_LED, v);
        Serial.printf("[ack] led %d\n", v);
    } else if (cmd.startsWith("G ")) {
        gainIdx = constrain(cmd.substring(2).toInt(), 0, 3);
        wr8(0x0F, GAIN_REGVAL[gainIdx]);
        Serial.printf("[ack] gain %d\n", gainIdx);
    }
}

void setup() {
    Serial.begin(460800);
    delay(500);
    analogWrite(PROBE_LED, 255);
    Wire.begin(PROBE_SDA, PROBE_SCL);
    Wire.setClock(100000);
    wr8(0x00, 0x01);
    delay(3);
    wr8(0x01, 0xD5);
    wr8(0x0F, GAIN_REGVAL[gainIdx]);
    wr8(0x00, 0x03);
}

void loop() {
    static String buf;
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') { handleCommand(buf); buf = ""; }
        else if (c != '\r') buf += c;
    }
    static uint32_t last = 0;
    if (millis() - last >= 120) {
        last = millis();
        Serial.printf("[tcs] C=%u R=%u G=%u B=%u\n",
                      rd16(0x14), rd16(0x16), rd16(0x18), rd16(0x1A));
    }
}
