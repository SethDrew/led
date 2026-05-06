/*
 * MIC_PROFILE — INMP441 raw-sample serial streamer (ESP32-C3)
 *
 * Streams 16-bit signed mono PCM at 16 kHz over USB CDC at 460800 baud.
 * Bandwidth: 16000 sps * 2 B = 32 KB/s; baud cap ~46 KB/s. Comfortable.
 *
 * Frame protocol (resync-safe):
 *   Every BLOCK_SAMPLES samples (default 160 = 10 ms), emit:
 *     [0xAA 0x55 0xA5 0x5A]      4-byte sync magic
 *     [uint16_t LE block_count]  rolls over freely; capture script tracks gaps
 *     [uint16_t LE n_samples]    number of int16 samples to follow (LE)
 *     [int16_t LE samples...]    n_samples * 2 bytes
 *   No checksum — capture script resyncs on next magic if a byte drops.
 *
 * Side channel: human-readable status lines start with '#' and are emitted
 * once per second (RMS sanity meter). Boot banner also '#'-prefixed so the
 * capture script can ignore '#' lines and pick out binary frames by magic.
 *
 * Reuses sender.cpp I2S config exactly (same pins, same DMA params).
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>

#define I2S_SCK     6
#define I2S_WS      5
#define I2S_SD      0

#define I2S_PORT       I2S_NUM_0
#define SAMPLE_RATE    16000
#define DMA_BUF_LEN    320
#define DMA_BUF_COUNT  4

#define BLOCK_SAMPLES  160        // 10 ms blocks at 16 kHz

static const uint8_t SYNC_MAGIC[4] = {0xAA, 0x55, 0xA5, 0x5A};

static uint16_t blockCounter = 0;
static uint64_t totalSamples = 0;
static double   sumSqSecond  = 0.0;
static uint32_t samplesInSecond = 0;
static uint32_t lastStatusMs = 0;

static int16_t  blockBuf[BLOCK_SAMPLES];

static void setupI2S() {
    i2s_config_t i2s_config = {
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
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num  = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_SD
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
}

static void emitBlock(const int16_t* samples, uint16_t n) {
    Serial.write(SYNC_MAGIC, 4);
    uint16_t bc = blockCounter++;
    Serial.write((uint8_t*)&bc, 2);
    Serial.write((uint8_t*)&n,  2);
    Serial.write((uint8_t*)samples, (size_t)n * 2);
}

void setup() {
    Serial.begin(460800);
    delay(800);
    setupI2S();

    Serial.printf("# [BOOT] fw=mic_profile sample_rate=%d bits=16 channels=1 baud=460800\n",
                  SAMPLE_RATE);
    Serial.printf("# [PROTOCOL] sync=AA55A55A then u16le block_count, u16le n_samples, n*int16le PCM\n");
    Serial.printf("# [BLOCK] samples_per_block=%d (%.1f ms)\n",
                  BLOCK_SAMPLES, 1000.0f * BLOCK_SAMPLES / SAMPLE_RATE);
    lastStatusMs = millis();
}

void loop() {
    int32_t raw[BLOCK_SAMPLES];
    size_t bytes_read = 0;
    esp_err_t err = i2s_read(I2S_PORT, raw, sizeof(raw), &bytes_read, portMAX_DELAY);
    if (err != ESP_OK || bytes_read == 0) return;

    int n = bytes_read / sizeof(int32_t);

    // INMP441 delivers 24-bit data left-justified in 32-bit word.
    // Shift right 16 to get a signed 16-bit sample (drops 8 LSBs of the 24).
    for (int i = 0; i < n; i++) {
        int32_t s24 = raw[i] >> 8;          // sign-extended 24-bit
        int16_t s16 = (int16_t)(s24 >> 8);  // top 16 bits of the 24
        blockBuf[i] = s16;
        double f = (double)s16;
        sumSqSecond += f * f;
    }
    samplesInSecond += n;
    totalSamples    += n;

    emitBlock(blockBuf, (uint16_t)n);

    uint32_t now = millis();
    if (now - lastStatusMs >= 1000) {
        double rms = samplesInSecond > 0
            ? sqrt(sumSqSecond / samplesInSecond) : 0.0;
        Serial.printf("# [RMS] %.1f  samples=%lu total=%llu\n",
                      rms, (unsigned long)samplesInSecond,
                      (unsigned long long)totalSamples);
        sumSqSecond = 0.0;
        samplesInSecond = 0;
        lastStatusMs = now;
    }
}
