/*
 * audio_recorder.cpp — lo-fi audio capture (TEST firmware).
 *
 * Temporarily pillages the bench-bulbs board to be a recorder: receives the
 * duck's AudioStreamPacketV1 waveform over ESP-NOW, writes it to LittleFS, and
 * dumps the file over serial as hex so a host can wrap a .wav. No LEDs, no
 * animations — this is its own [env:audio_recorder] / source file; bench.cpp
 * is untouched. Flash for a capture, reflash [env:bench] to restore the rig.
 *
 * Channel is HARDCODED to 1 (no cuteplant SSID scan) so it can't tune away
 * from the duck.
 *
 * Serial protocol @ 460800:
 *   R  begin recording (truncates /audio.raw)
 *   S  stop recording (flush + close)
 *   D  dump /audio.raw as framed hex for the host
 *   I  info (size, packets, drops, overflow)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <LittleFS.h>
#include "audio_stream_packet_v1.h"

#define FIXED_CHANNEL 1
#define REC_PATH      "/audio.raw"
#define SAMPLE_RATE_HZ 8000          // matches AUDIO_STREAM_RATE_8K

// ── SPSC ring buffer (recv callback = producer, loop = consumer) ──
// 16 KB ≈ 2 s of 8 kB/s audio — absorbs LittleFS write stalls.
#define RING_SIZE  16384             // must be power of two
#define RING_MASK  (RING_SIZE - 1)
static uint8_t ring[RING_SIZE];
static volatile uint32_t ringHead = 0;   // producer writes
static volatile uint32_t ringTail = 0;   // consumer reads

static volatile bool     recording   = false;
static volatile uint32_t pktCount    = 0;   // packets accepted while recording
static volatile uint32_t dropCount   = 0;   // packets missed (seq gaps)
static volatile uint32_t overflowCnt = 0;   // bytes lost to ring overflow
static volatile uint32_t lastSeq     = 0;
static volatile bool     haveSeq     = false;

static File recFile;

// Free-running counters: unsigned (head - tail) is the byte count and wraps
// correctly at 2^32 as long as the fill never exceeds RING_SIZE.
static inline uint32_t ringUsed() { return ringHead - ringTail; }

// Push bytes into the ring; count any that don't fit as overflow.
static void ringPush(const uint8_t *p, uint32_t len, uint8_t fillVal, bool isFill) {
    for (uint32_t i = 0; i < len; i++) {
        if ((ringHead - ringTail) >= RING_SIZE) { overflowCnt++; return; }
        ring[ringHead & RING_MASK] = isFill ? fillVal : p[i];
        ringHead++;
    }
}

void onEspNowReceive(const uint8_t * /*mac*/, const uint8_t *data, int len) {
    if (!recording) return;
    if (len != (int)sizeof(AudioStreamPacketV1)) return;   // size-dispatch

    const AudioStreamPacketV1 *pkt = (const AudioStreamPacketV1 *)data;
    uint16_t seq = pkt->seq;

    // Gap-fill with silence (0x80) so the recording stays time-accurate.
    // Only fill modest gaps; a big jump means start-of-stream or a long
    // dropout — resync instead of injecting seconds of silence.
    if (haveSeq) {
        uint16_t expected = (uint16_t)(lastSeq + 1);
        uint16_t gap = (uint16_t)(seq - expected);
        if (gap > 0 && gap <= 50) {
            dropCount += gap;
            ringPush(nullptr, (uint32_t)gap * AUDIO_STREAM_SAMPLES, 0x80, true);
        }
    }
    lastSeq = seq;
    haveSeq = true;

    uint8_t n = pkt->n;
    if (n > AUDIO_STREAM_SAMPLES) n = AUDIO_STREAM_SAMPLES;
    ringPush(pkt->samples, n, 0, false);
    pktCount++;
}

static void startRecording() {
    if (recording) { Serial.println("[REC] already recording"); return; }
    recFile = LittleFS.open(REC_PATH, "w");
    if (!recFile) { Serial.println("[REC] ERROR: cannot open file"); return; }
    ringHead = ringTail = 0;
    pktCount = dropCount = overflowCnt = 0;
    haveSeq = false;
    recording = true;
    Serial.println("[REC] start");
}

static void stopRecording() {
    if (!recording) { Serial.println("[REC] not recording"); return; }
    recording = false;
    delay(50);                       // let in-flight callback finish
    // drain whatever is left in the ring
    while (ringUsed() > 0) {
        uint8_t b = ring[ringTail & RING_MASK];
        ringTail = (ringTail + 1) & 0xFFFFFFFF;
        recFile.write(b);
    }
    recFile.flush();
    size_t sz = recFile.size();
    recFile.close();
    Serial.printf("[REC] stop  bytes=%u pkts=%lu drops=%lu overflow=%lu  (%.1f s)\n",
                  (unsigned)sz, (unsigned long)pktCount, (unsigned long)dropCount,
                  (unsigned long)overflowCnt, (float)sz / SAMPLE_RATE_HZ);
}

static void dumpFile() {
    File f = LittleFS.open(REC_PATH, "r");
    if (!f) { Serial.println("[DUMP] ERROR: no file"); return; }
    size_t sz = f.size();
    Serial.printf("[DUMP] start size=%u rate=%d bits=8 enc=u8\n", (unsigned)sz, SAMPLE_RATE_HZ);
    static const char hexd[] = "0123456789abcdef";
    char line[129];                  // 64 bytes -> 128 hex chars + NUL
    uint8_t rd[64];
    while (true) {
        int got = f.read(rd, sizeof(rd));
        if (got <= 0) break;
        int p = 0;
        for (int i = 0; i < got; i++) {
            line[p++] = hexd[rd[i] >> 4];
            line[p++] = hexd[rd[i] & 0x0F];
        }
        line[p] = '\0';
        Serial.println(line);
    }
    f.close();
    Serial.println("[DUMP] end");
}

static void info() {
    size_t sz = 0;
    if (LittleFS.exists(REC_PATH)) {
        File f = LittleFS.open(REC_PATH, "r");
        if (f) { sz = f.size(); f.close(); }
    }
    Serial.printf("[INFO] recording=%d file_bytes=%u pkts=%lu drops=%lu overflow=%lu ring=%lu/%d\n",
                  recording ? 1 : 0, (unsigned)sz, (unsigned long)pktCount,
                  (unsigned long)dropCount, (unsigned long)overflowCnt,
                  (unsigned long)ringUsed(), RING_SIZE);
}

void setup() {
    Serial.begin(460800);
    delay(300);

    if (!LittleFS.begin(true)) {
        Serial.println("[BOOT] LittleFS mount FAILED");
    }

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);

    if (esp_now_init() != ESP_OK) {
        Serial.println("[BOOT] ESP-NOW init FAILED");
        while (1) delay(1000);
    }
    esp_now_register_recv_cb(onEspNowReceive);

    Serial.printf("[BOOT] role=recorder MAC=%s ch=%d  (R=rec S=stop D=dump I=info)\n",
                  WiFi.macAddress().c_str(), WiFi.channel());
}

void loop() {
    // Drain ring -> file while recording.
    if (recording) {
        uint8_t chunk[256];
        int got = 0;
        while (ringUsed() > 0 && got < (int)sizeof(chunk)) {
            chunk[got++] = ring[ringTail & RING_MASK];
            ringTail = (ringTail + 1) & 0xFFFFFFFF;
        }
        if (got > 0) recFile.write(chunk, got);
    }

    // Serial command handling.
    if (Serial.available()) {
        char c = (char)Serial.read();
        switch (c) {
            case 'R': case 'r': startRecording(); break;
            case 'S': case 's': stopRecording();  break;
            case 'D': case 'd': dumpFile();        break;
            case 'I': case 'i': info();            break;
            default: break;                        // ignore CR/LF/noise
        }
    }
}
