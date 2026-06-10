/*
 * dac_test — 3.5mm jack / GPIO25 DAC bring-up test (additive, not production).
 *
 * Plays a repeating A-major arpeggio (A4 / C#5 / E5) followed by a silence gap
 * through the internal DAC on GPIO25, using the EXACT I2S config the real
 * tree-of-record playback path uses (I2S_NUM_0, DAC built-in, 8 kHz, 16-bit,
 * both channels enabled). This isolates the analog wiring — jack, coupling
 * cap, GPIO25 — from the whole record/playback pipeline.
 *
 *   Clean tones      → jack + cap + DAC + GPIO25 all good.
 *   Tone on GPIO26   → both DAC channels are fed; the jack works on either pin.
 *   Hum/buzz only    → grounding / sleeve wiring issue.
 *   Nothing          → no GPIO25 connection, or amp not on line level.
 *   Silence gap noisy → noise floor (the coupling-cap bias settling, usually ok).
 *
 * Samples are 8-bit UNSIGNED centered at 128 (silence = 128), matching the
 * recorded-audio format, placed in the high byte of each 16-bit I2S word and
 * duplicated L/R — same as playbackAudioTick().
 *
 * Build/flash:
 *   cd festicorn/tree-of-record
 *   ../../.venv/bin/pio run -e dac_test -t upload   # add --upload-port if needed
 *   ../../.venv/bin/pio device monitor -e dac_test
 * Restore the real firmware with: pio run -e biolum-a -t upload
 */
#include <Arduino.h>
#include <math.h>
#include <driver/i2s.h>

static const int      SAMPLE_RATE = 8000;     // matches REC_AUDIO_RATE
static const i2s_port_t PORT      = I2S_NUM_0;

// Arpeggio: note frequency (Hz) then a 0 = silence. Each step is STEP_MS long.
static const float  NOTES[]  = { 440.0f, 554.37f, 659.25f, 0.0f };
static const int    N_NOTES  = sizeof(NOTES) / sizeof(NOTES[0]);
static const int    STEP_MS  = 400;
static const float  AMPLITUDE = 100.0f;       // 0..127 around the 128 midpoint

static void dacInstall() {
    i2s_config_t cfg = {};
    cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_DAC_BUILT_IN);
    cfg.sample_rate = SAMPLE_RATE;
    cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
    cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
    cfg.communication_format = I2S_COMM_FORMAT_STAND_MSB;
    cfg.intr_alloc_flags = 0;
    cfg.dma_buf_count = 8;
    cfg.dma_buf_len = 256;
    cfg.use_apll = false;
    if (i2s_driver_install(PORT, &cfg, 0, NULL) != ESP_OK) {
        Serial.println("[DAC] i2s install FAILED");
        return;
    }
    i2s_set_dac_mode(I2S_DAC_CHANNEL_BOTH_EN);   // GPIO25 + GPIO26 both driven
    i2s_zero_dma_buffer(PORT);
    Serial.println("[DAC] installed on GPIO25/26, 8 kHz");
}

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println("\n[dac_test] A-major arpeggio on GPIO25. Plug in & listen.");
    dacInstall();
}

void loop() {
    static float    phase = 0.0f;
    static uint16_t dbuf[256 * 2];               // L/R interleaved

    for (int n = 0; n < N_NOTES; n++) {
        float freq = NOTES[n];
        Serial.printf("[DAC] step %d: %s\n", n,
                      freq > 0 ? "tone" : "silence");
        int totalSamples = SAMPLE_RATE * STEP_MS / 1000;
        float dphi = (freq > 0.0f) ? (2.0f * (float)M_PI * freq / SAMPLE_RATE) : 0.0f;

        int remaining = totalSamples;
        while (remaining > 0) {
            int block = remaining > 256 ? 256 : remaining;
            for (int i = 0; i < block; i++) {
                uint8_t s;
                if (freq > 0.0f) {
                    s = (uint8_t)(128.0f + AMPLITUDE * sinf(phase));
                    phase += dphi;
                    if (phase > 2.0f * (float)M_PI) phase -= 2.0f * (float)M_PI;
                } else {
                    s = 128;                     // silence = midscale
                }
                uint16_t v = (uint16_t)s << 8;   // 8-bit value in high byte
                dbuf[2 * i] = v;
                dbuf[2 * i + 1] = v;
            }
            size_t wrote = 0;
            i2s_write(PORT, dbuf, (size_t)block * 2 * sizeof(uint16_t),
                      &wrote, portMAX_DELAY);
            remaining -= block;
        }
    }
}
