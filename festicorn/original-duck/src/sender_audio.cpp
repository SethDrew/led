/*
 * sender_audio.cpp — lo-fi audio WAVEFORM streamer (TEST firmware).
 *
 * Temporarily repurposes the duck board to stream what its INMP441 hears as
 * playable 8 kHz / 8-bit PCM over ESP-NOW, so we can pull a .wav off the bench
 * recorder and hear the mic. Audio only — no IMU, no telemetry trio.
 *
 * This is ADDITIVE: it is its own [env:sender_audio] / source file. The frozen
 * sender.cpp and the deployed sender_v2.cpp are untouched. Flash this for a
 * capture, then reflash sender_v2 to restore the totem.
 *
 *   Duck pinout (same as sender_v2): SCK=6, WS=5, SD=0 (INMP441).
 *   ESP-NOW broadcast on channel 1 (matches the bench recorder + tree).
 *
 * Pipeline: I2S 16 kHz 32-bit -> (raw>>8) 24-bit -> ÷2 average-decimate to
 * 8 kHz -> one-pole DC-removal HPF -> linear 8-bit quantize (silence=128)
 * -> 200-sample AudioStreamPacketV1 @ 40 packets/s (~8.2 kB/s, <5% airtime).
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "audio_stream_packet_v1.h"

// ── Pins / I2S (ported verbatim from sender_v2.cpp) ───────────────
#define I2S_SCK       6
#define I2S_WS        5
#define I2S_SD        0
#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   16000
#define DMA_BUF_LEN   320
#define DMA_BUF_COUNT 4

#define FIXED_CHANNEL 1

// ── The one by-ear knob ───────────────────────────────────────────
// Quantize: byte = clamp((ac >> QUANT_SHIFT) + 128). The AC sample lives in
// the same 24-bit (raw>>8) domain as our RMS calibration (silent room ~2k,
// speech-at-mic RMS up to ~156k; instantaneous peaks run a few× the RMS).
// Lower QUANT_SHIFT = louder & more clipping, higher = quieter & cleaner.
// Start here, listen to the first clip, then nudge ±1.
#define QUANT_SHIFT   11

// One-pole DC-removal HPF: dc += (s - dc) >> DC_SHIFT, ac = s - dc.
// >>10 @ 8 kHz ≈ 1024-sample tau ≈ 128 ms ≈ 1.2 Hz cutoff — kills the mic's
// DC bias, keeps voice. (DC removal is mandatory: without it everything
// rails to one quantizer extreme.)
#define DC_SHIFT      10

static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

static AudioStreamPacketV1 pkt;
static uint8_t  fill = 0;
static uint16_t seqCounter = 0;
static uint32_t sendErrs = 0;
static uint32_t pktsSent = 0;

// decimation + DC-removal state
static int32_t dcEma   = 0;
static int32_t held    = 0;
static bool    havePrev = false;

// ── I2S setup (identical to sender_v2) ────────────────────────────
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
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
}

static void onSent(const uint8_t * /*mac*/, esp_now_send_status_t status) {
    if (status != ESP_NOW_SEND_SUCCESS) sendErrs++;
}

static void emitPacket() {
    pkt.seq       = seqCounter++;
    pkt.rate_code = AUDIO_STREAM_RATE_8K;
    pkt.n         = fill;
    if (esp_now_send(broadcastAddr, (uint8_t*)&pkt, sizeof(pkt)) != ESP_OK) sendErrs++;
    fill = 0;
    pktsSent++;
}

void setup() {
    Serial.begin(460800);
    delay(300);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setTxPower(WIFI_POWER_8_5dBm);
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init FAILED");
        while (1) delay(1000);
    }
    esp_now_register_send_cb(onSent);

    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, broadcastAddr, 6);
    peer.channel = 0;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    setupI2S();

    Serial.printf("[BOOT] role=duck fw=sender_audio MAC=%s ch=%d\n",
                  WiFi.macAddress().c_str(), WiFi.channel());
    Serial.printf("streaming 8kHz/8-bit PCM, %d samples/pkt, QUANT_SHIFT=%d\n",
                  AUDIO_STREAM_SAMPLES, QUANT_SHIFT);
}

void loop() {
    static uint32_t lastLog = 0;
    int32_t buf[DMA_BUF_LEN];
    size_t  bytesRead = 0;

    // Blocking read paces the loop to the 16 kHz DMA (no busy-spin).
    if (i2s_read(I2S_PORT, buf, sizeof(buf), &bytesRead, portMAX_DELAY) != ESP_OK)
        return;

    int num = bytesRead / sizeof(int32_t);
    for (int i = 0; i < num; i++) {
        int32_t s = buf[i] >> 8;            // 24-bit domain (matches RMS calib)

        // ÷2 average-decimate 16 kHz -> 8 kHz (also a crude anti-alias LPF).
        if (!havePrev) { held = s; havePrev = true; continue; }
        int32_t avg = (held + s) >> 1;
        havePrev = false;

        // One-pole DC removal.
        dcEma += (avg - dcEma) >> DC_SHIFT;
        int32_t ac = avg - dcEma;

        // Linear 8-bit quantize, silence = 128.
        int32_t q = (ac >> QUANT_SHIFT) + 128;
        if (q < 0)   q = 0;
        if (q > 255) q = 255;

        pkt.samples[fill++] = (uint8_t)q;
        if (fill >= AUDIO_STREAM_SAMPLES) emitPacket();
    }

    uint32_t now = millis();
    if (now - lastLog >= 1000) {
        lastLog = now;
        Serial.printf("pkts=%lu errs=%lu ch=%d\n",
                      (unsigned long)pktsSent, (unsigned long)sendErrs, WiFi.channel());
    }
}
