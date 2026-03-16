// Bluetooth A2DP Audio Streamer
// Receives raw 16-bit mono PCM at 22050Hz over serial.
#include <Arduino.h>
#include "BluetoothA2DPSource.h"

#define LED_PIN 2
#define SERIAL_BAUD 921600
#define RING_SIZE 8192
#define PREBUF_SAMPLES 2048

const char* SPEAKER_NAME = "JBL Clip 3";

BluetoothA2DPSource a2dp;

int16_t ring[RING_SIZE];
volatile int wr = 0;
volatile int rd = 0;
volatile bool playing = false;

int ring_count() {
    int d = wr - rd;
    return d < 0 ? d + RING_SIZE : d;
}

int32_t audio_cb(Frame* data, int32_t count) {
    for (int i = 0; i < count; i++) {
        if (playing && ring_count() > 0) {
            int16_t s = ring[rd];
            rd = (rd + 1) % RING_SIZE;
            // Duplicate sample: 22050 -> 44100
            data[i].channel1 = s;
            data[i].channel2 = s;
            i++;
            if (i < count) {
                data[i].channel1 = s;
                data[i].channel2 = s;
            }
        } else {
            data[i].channel1 = 0;
            data[i].channel2 = 0;
        }
    }
    return count;
}

void setup() {
    Serial.setRxBufferSize(4096);
    Serial.begin(SERIAL_BAUD);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);
    delay(2000);
    Serial.println("=== BT Streamer ===");
    Serial.flush();

    a2dp.set_local_name("LED Weld");  // ESP32's BT name
    a2dp.set_auto_reconnect(true);
    a2dp.set_volume(255);
    a2dp.start(SPEAKER_NAME, audio_cb);

    Serial.println("READY");
    Serial.flush();
    digitalWrite(LED_PIN, LOW);
}

void loop() {
    // Bulk read serial into ring buffer
    int avail = Serial.available();
    if (avail >= 2) {
        static uint8_t buf[512];
        if (avail > (int)sizeof(buf)) avail = sizeof(buf);
        avail &= ~1;  // even bytes only
        int n = Serial.readBytes(buf, avail);
        for (int i = 0; i + 1 < n; i += 2) {
            int16_t sample = (int16_t)(buf[i] | (buf[i+1] << 8));
            int next = (wr + 1) % RING_SIZE;
            if (next != rd) {
                ring[wr] = sample;
                wr = next;
            }
        }
    }

    if (!playing && ring_count() >= PREBUF_SAMPLES) {
        playing = true;
    }

    // Print speaker MAC once after connection
    static bool mac_printed = false;
    if (!mac_printed && a2dp.is_connected()) {
        const uint8_t* peer = (const uint8_t*)a2dp.get_last_peer_address();
        Serial.printf("SPEAKER_MAC=%02X:%02X:%02X:%02X:%02X:%02X\n",
            peer[0], peer[1], peer[2], peer[3], peer[4], peer[5]);
        Serial.flush();
        mac_printed = true;
    }

    yield();
}
