/*
 * wav_play — play the embedded raw capture through GPIO25 DAC (test firmware).
 *
 * Loops audio_capture.wav (the no-boost capture, embedded as wav_samples.h)
 * verbatim through the same I2S DAC path as real playback: I2S_NUM_0, DAC
 * built-in, 8 kHz, 16-bit, both channels. No gain — plays exactly what was
 * recorded, so you hear the true lo-fi character on real audio (vs the harsh
 * test tones).
 *
 * Build/flash:
 *   cd festicorn/tree-of-record
 *   ../../.venv/bin/pio run -e wav_play -t upload --upload-port <port>
 * Restore real firmware: pio run -e biolum-a -t upload
 */
#include <Arduino.h>
#include <driver/i2s.h>
#include "wav_samples.h"

static const int        SAMPLE_RATE = 8000;   // true content rate
// Legacy arduino-esp32 I2S→built-in-DAC has an 8-bit mclk_div overflow bug
// (arduino-esp32 #5938, fixed in core 2.0.3): low configured sample rates
// compute a divisor >255 that truncates, so 8 kHz plays chipmunked at an
// arbitrary fast pitch. Rates ≳22 kHz don't overflow and are honored ~1:1.
// Fix without bumping the core: run the DAC at an honored rate that's an
// integer multiple of 8 kHz and repeat each source sample UPSAMPLE times
// (zero-order hold). 24000 = 3×8000; measured content rate = 8001 Hz.
static const int        UPSAMPLE    = 3;
static const int        DAC_RATE    = SAMPLE_RATE * UPSAMPLE;   // 24000, honored
static const i2s_port_t PORT        = I2S_NUM_0;

static void dacInstall() {
    i2s_config_t cfg = {};
    cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_DAC_BUILT_IN);
    cfg.sample_rate = DAC_RATE;
    cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
    cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
    cfg.communication_format = I2S_COMM_FORMAT_STAND_MSB;
    cfg.intr_alloc_flags = 0;
    cfg.dma_buf_count = 8;
    cfg.dma_buf_len = 256;
    cfg.use_apll = false;
    if (i2s_driver_install(PORT, &cfg, 0, NULL) != ESP_OK) {
        Serial.println("[WAV] i2s install FAILED");
        return;
    }
    i2s_set_dac_mode(I2S_DAC_CHANNEL_BOTH_EN);
    i2s_zero_dma_buffer(PORT);
    Serial.println("[WAV] DAC on GPIO25/26, 8 kHz");
}

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.printf("\n[wav_play] looping %lu samples (%.1fs), no boost, on GPIO25.\n",
                  (unsigned long)WAV_LEN, WAV_LEN / (float)SAMPLE_RATE);
    dacInstall();
}

void loop() {
    // Each source sample expands to UPSAMPLE frames, each frame = L+R words.
    static const int SRC_BLK = 128;
    static uint16_t dbuf[SRC_BLK * 3 * 2];   // sized for UPSAMPLE up to 3
    uint32_t pos = 0;
    Serial.println("[WAV] play");
    while (pos < WAV_LEN) {
        int block = (WAV_LEN - pos) > SRC_BLK ? SRC_BLK : (int)(WAV_LEN - pos);
        int w = 0;
        for (int i = 0; i < block; i++) {
            uint16_t v = (uint16_t)WAV_SAMPLES[pos + i] << 8;   // 8-bit in high byte
            for (int u = 0; u < UPSAMPLE; u++) {                // zero-order hold ×UPSAMPLE
                dbuf[w++] = v;   // L
                dbuf[w++] = v;   // R
            }
        }
        size_t wrote = 0;
        i2s_write(PORT, dbuf, (size_t)w * sizeof(uint16_t), &wrote, portMAX_DELAY);
        pos += block;
    }
}
