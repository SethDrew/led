/*
 * RAW AUDIO SENDER — ESP32-C3 streams INMP441 samples over USB CDC serial.
 *
 * Single-purpose validation firmware: continuously read I2S DMA, take the top
 * 16 bits of each 24-bit sample, and write a contiguous int16 LE stream to
 * USB serial @ 460800 baud. No IMU, no ESP-NOW, no framing — host treats the
 * incoming bytes as raw 16 kHz mono PCM.
 *
 * 16 kHz * 2 B = 32 kB/s payload — well within USB CDC throughput at 460800.
 *
 * Hardware: ESP32-C3 super-mini, INMP441 (I2S SCK=6 WS=5 SD=0).
 */

#include <Arduino.h>
#include <driver/i2s.h>

#define I2S_SCK  6
#define I2S_WS   5
#define I2S_SD   0

#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000
#define DMA_BUF_LEN   320      // ~20 ms @ 16 kHz
#define DMA_BUF_COUNT 4

#define STATUS_LED_PIN  8
#define STATUS_LED_ON   LOW
#define STATUS_LED_OFF  HIGH

static void setupI2S() {
    i2s_config_t cfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = DMA_BUF_COUNT,
        .dma_buf_len = DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    i2s_pin_config_t pins = {
        .bck_io_num = I2S_SCK,
        .ws_io_num  = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_SD
    };
    i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
    i2s_set_pin(I2S_PORT, &pins);
}

void setup() {
    Serial.begin(460800);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, STATUS_LED_OFF);
    setupI2S();
    delay(200);
}

void loop() {
    static int32_t  rxBuf[DMA_BUF_LEN];
    static int16_t  txBuf[DMA_BUF_LEN];
    size_t bytes_read = 0;

    if (i2s_read(I2S_PORT, rxBuf, sizeof(rxBuf), &bytes_read, portMAX_DELAY) == ESP_OK
        && bytes_read > 0) {
        int n = bytes_read / sizeof(int32_t);
        // INMP441 packs 24-bit data left-justified in a 32-bit slot.
        // Right-shift by 16 to take the most-significant 16 bits (signed).
        for (int i = 0; i < n; i++) {
            txBuf[i] = (int16_t)(rxBuf[i] >> 16);
        }
        Serial.write((const uint8_t *)txBuf, n * sizeof(int16_t));
    }

    static uint32_t lastToggle = 0;
    static bool ledOn = false;
    uint32_t now = millis();
    if (now - lastToggle >= 250) {
        lastToggle = now;
        ledOn = !ledOn;
        digitalWrite(STATUS_LED_PIN, ledOn ? STATUS_LED_ON : STATUS_LED_OFF);
    }
}
