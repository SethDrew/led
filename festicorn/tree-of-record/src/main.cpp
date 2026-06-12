/*
 * TREE-OF-RECORD — ambient bloom on all 6 strips, driven by a BS-26.
 *
 * ESP32-D0WD-V3, 6 × WS2812B RGB strips on GPIO 4, 15, 17, 5, 18, 19,
 * 100 LEDs each. No senders, no motion — pure ambient breathing bloom
 * (ported from biolum board B), modulated by a BS-26 console:
 *
 *   tens (10-pos knob) → brightness; 0 = fully off
 *   ones (0–11)        → speed (12 steps, log-spaced 0.2–8.0)
 *   decouter (0–11)    → effect select, grouped: 0–3 audio field effects
 *                        (bloom, fire, heat, worley), 4–6 audio particle
 *                        effects (gravity, sparkle, creatures), 7–8 ambient
 *                        particles (polycule, leaf-wind), 9–11 passive
 *                        (light-through, nebula, rainbow)
 *   hundreds (0–4)     → recording slot select (0–4)
 *   decmid (0–9)       → hue        36° rotation steps (full wheel)
 *   decinner (0–9)     → saturation 5=original, 0=boosted to full, 9=white
 *   AC switch          → record (toggle: start/stop recording to selected slot)
 *   DC switch          → playback (toggle: start/stop playback from selected slot)
 *
 * BS-26 live → knobs control. No signal → serial PARAMS control.
 * BS-26 state arrives as JSON over ESP-NOW broadcast on channel 1.
 *
 * ── INVARIANTS (hold these when editing any effect) ──────────────
 * 1. Two clocks. Effects receive fxDt (= dt × speed knob × SPEED_SCALE) for
 *    motion, fades, spawning. Anything that REACTS to audio or sensors —
 *    envelopes, onset cooldowns, energy EMAs — uses real gDt. The ones knob
 *    shapes motion, never reaction latency.
 * 2. Gamma goes on a scalar brightness (fastGamma24), which then scales all
 *    three channels uniformly. Never gamma R/G/B individually — it shifts
 *    hue on mixed colors as they dim.
 * 3. New effects take color from oklchVarL (perceptually tuned LUT). The
 *    hand-RGB palettes in fire/nebula/light-through/leaf-wind/creatures are
 *    deliberate per-effect artistic choices, not the default.
 * 4. All output goes through setPixelScaled — that's where the hue/sat
 *    knobs, renderBrightness, and the floor policy live. Bloom is the one
 *    sanctioned exception (rotates its own endpoints via applyPalette).
 * 5. PARAMS[] exposes values tuned live in the field; constants tuned once
 *    stay #defines. Don't expose by reflex.
 * 6. All decay/smoothing is dt-based (festicorn/CLAUDE.md). Legacy
 *    fastDecay(rate, dt×30) sites are dt-correct — the ×30 is a reference
 *    frame rate baked into the rate constant, not an FPS assumption to fix.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_random.h>
#include <NeoPixelBus.h>
#include <ArduinoJson.h>
#include <SPIFFS.h>
#include <math.h>
#include <fast_math.h>
#include <oklch_lut.h>
#include <driver/i2s.h>
#include <v1_packet.h>
#include <gyro_packet_v1.h>
#include <audio_packet_v1.h>
#include <audio_stream_packet_v1.h>

// ── Strip layout ─────────────────────────────────────────────────
#ifndef LEDS_PER_STRIP
#define LEDS_PER_STRIP 100
#endif

static const uint8_t NUM_STRIPS = 6;

// ── Bloom parameters (runtime-tunable via operator) ─────────────
static float bloomBrightnessCap  = 1.0f;
static float bloomBufferDrain     = 15.0f;
static float bloomFlashDecayRate  = 3.0f;
static float bloomBreathFloor     = 0.32f;   // glow trough (pulses, doesn't blink)
// Audio-energy overdrive: audioEnergy5s drives a per-pixel boost ADDED after the
// brightness knob/cap, so loud audio punches bright pixels THROUGH the cap toward
// saturation (up to bloomAudioCeil ≈ 2–3×) even when the brightness knob is low,
// while below-mean pixels are pushed dark. Knob sets the quiet ambient level.
static float bloomStretchGain     = 2.5f;   // overdrive level at full energy (× cap)
static float bloomAudioCeil       = 3.0f;   // max level; ×baseColor saturates at 255

#define BLOOM_BREATH_MIN_PERIOD 3.0f
#define BLOOM_BREATH_MAX_PERIOD 8.0f
#define BLOOM_BREATH_MIN_PEAK   0.65f
#define BLOOM_BREATH_MAX_PEAK   1.00f

#define BLOOM_FLASH_R  200.0f
#define BLOOM_FLASH_G  120.0f
#define BLOOM_FLASH_B  255.0f

#define BLOOM_HUE_DRIFT_MIN  (1.0f / 45.0f)
#define BLOOM_HUE_DRIFT_MAX  (1.0f / 15.0f)

static float bloomDormancyMin  = 3.0f;
static float bloomDormancyMax  = 7.0f;
static float bloomDormancyFrac = 0.0f;

// AC switch on the BS-26 → record toggle. DC switch → playback toggle.
// acOn retained for the "tens=0 means off" path but no longer driven by AC.
static bool acOn = true;

// ── Recording system ─────────────────────────────────────────────
// Up to 5 slots of 30s input recordings stored in SPIFFS.
// Each slot: a small header (RecFileHeader) followed by a packed array of
// RecFrame structs at capture framerate (~60fps). Each frame stores the knob/
// effect params plus the filtered audio drive bytes so silent replay stays
// faithful. RecFrame is 8 bytes; 1800 frames ≈ 14.4 KB per slot.
#define REC_MAX_SLOTS     5
#define REC_MAX_FRAMES    1800   // 30s × 60fps
#define REC_FPS           60

enum RecState { REC_IDLE, REC_RECORDING, REC_PLAYING };
static RecState recState = REC_IDLE;
static uint8_t  recSlot = 0;          // selected via hundreds knob (0–4)
// Maintained on/off switches (level semantics): switch position = state.
// A "latch" disarms the switch after its action fires so the 30 s record cap
// can't instantly restart while the switch is still held — you must flip the
// switch off (re-arm) to record/play again.
static bool     recAcLatch = false;    // AC (record) switch consumed-while-high
static bool     recDcLatch = false;    // DC (playback) switch consumed-while-high

struct RecFrame {
    uint8_t tens, ones, hundreds, decouter, decmid, decinner;
    // Filtered audio drive captured with the frame so playback is faithful in
    // silence: the AudioPacketV1 companded RMS bytes (25 Hz, sqrt-companded).
    // On playback these are re-derived through the same floor/ceiling/onset
    // path as live packets (see audioDeriveFeatures). 0/0 = muted or stale.
    uint8_t rms_mean, rms_max;
};

// On-disk /rec_N.bin format: a small header then a packed array of RecFrame.
//   magic   "TRR1" — distinguishes versioned files from legacy headerless ones
//   version format revision (1 = first with audio bytes)
//   frameSize sizeof(RecFrame) at write time, so a future field add still loads
// Legacy files (pre-header, 6-byte frames) lack the magic; they are loaded as
// v0 with rms bytes = 0 (audio renders silent on replay) rather than rejected.
struct RecFileHeader {
    char     magic[4];   // 'T','R','R','1'
    uint8_t  version;    // current REC_FILE_VERSION
    uint8_t  frameSize;  // sizeof(RecFrame) at write time
    uint16_t reserved;   // 0, padding / forward use
};
static const char REC_MAGIC[4]      = { 'T', 'R', 'R', '1' };
#define REC_FILE_VERSION  1
#define REC_LEGACY_FRAME_SIZE 6   // old headerless format: 6-byte RecFrame

static RecFrame* recBuffer = nullptr;  // heap-allocated on first use
static uint32_t  recFrameCount = 0;    // frames written/total in buffer
static uint32_t  recPlayIdx = 0;       // current playback position
static float     recFrameAcc = 0.0f;   // record: dt accumulator for fixed-rate capture
static float     recPlayClock = 0.0f;  // playback: wall-clock fallback when no audio
static bool      recSlotHasData[REC_MAX_SLOTS] = {};

// ── Audio waveform record/playback (8 kHz/8-bit, /rec_N.audio) ──
// The duck (sender_v3) streams AudioStreamPacketV1 only while a slot records.
// We capture it to a parallel file, then on playback feed it to the DAC
// (GPIO25 via I2S-DMA) and re-derive fxEnergy/fxOnset from the same samples.
#define REC_AUDIO_RATE   8000
#define REC_AUDIO_MAX    (REC_AUDIO_RATE * 30)   // 30 s cap = 240000 bytes
#define WF_QUANT_SHIFT   11                       // must match sender_v3 AUDIO_QUANT_SHIFT
// Legacy arduino-esp32 I2S→built-in-DAC has an 8-bit mclk_div overflow bug
// (arduino-esp32 #5938, fixed in core 2.0.3): configuring sub-~22 kHz rates
// truncates the divisor and plays chipmunked. Rates ≳22 kHz are honored ~1:1,
// so we clock the DAC at REC_AUDIO_RATE×WF_DAC_UPSAMPLE and repeat each source
// sample that many times (zero-order hold). 24 kHz measured = 8001 Hz content.
#define WF_DAC_UPSAMPLE  3                        // 8 kHz × 3 = 24 kHz DAC clock
#define WFR_SIZE         4096                      // RX ring (power of two), ~0.5 s
                                                   // drained every loop (~16 ms), so
                                                   // 0.5 s is ample slack vs 8 kHz in.
#define WFR_MASK         (WFR_SIZE - 1)
static uint8_t   wfRing[WFR_SIZE];
static volatile uint32_t wfHead = 0, wfTail = 0;  // SPSC: onReceive→loop
static File      recAudioFile;
static bool      recAudioOpen  = false;
static uint32_t  recAudioBytes = 0;
static bool      recSlotHasAudio[REC_MAX_SLOTS] = {};
static File      playAudioFile;
static bool      playAudioOpen = false;
static bool      dacInstalled  = false;

static void recInit() {
    if (!SPIFFS.begin(true)) {
        Serial.println("[REC] SPIFFS mount failed");
        return;
    }
    Serial.printf("[REC] SPIFFS ready, %lu bytes free\n",
                  (unsigned long)SPIFFS.totalBytes() - SPIFFS.usedBytes());
    for (int i = 0; i < REC_MAX_SLOTS; i++) {
        char path[24];
        uint32_t binSz = 0, audSz = 0;
        snprintf(path, sizeof(path), "/rec_%d.bin", i);
        recSlotHasData[i] = SPIFFS.exists(path);
        if (recSlotHasData[i]) { File f = SPIFFS.open(path, "r"); if (f) { binSz = f.size(); f.close(); } }
        snprintf(path, sizeof(path), "/rec_%d.audio", i);
        recSlotHasAudio[i] = SPIFFS.exists(path);
        if (recSlotHasAudio[i]) { File f = SPIFFS.open(path, "r"); if (f) { audSz = f.size(); f.close(); } }
        if (recSlotHasData[i] || recSlotHasAudio[i])
            Serial.printf("[REC]   slot %d: frames=%lu B (%.1fs)  audio=%lu B (%.1fs)\n",
                          i, (unsigned long)binSz, binSz / (8.0f * 60.0f),
                          (unsigned long)audSz, audSz / (float)REC_AUDIO_RATE);
    }
}

static bool recAllocBuffer() {
    if (recBuffer) return true;
    recBuffer = (RecFrame*)malloc(REC_MAX_FRAMES * sizeof(RecFrame));
    if (!recBuffer) {
        Serial.println("[REC] buffer alloc failed");
        return false;
    }
    return true;
}

// ── Waveform helpers ─────────────────────────────────────────────
static void audioDeriveFeaturesRms(float rmsMean, float rmsMax, float dt);  // fwd

static void wfRingReset() { wfHead = wfTail = 0; }
static void wfRingPush(const uint8_t *s, uint32_t n) {       // onReceive (producer)
    for (uint32_t i = 0; i < n; i++) {
        if ((wfHead - wfTail) >= WFR_SIZE) return;           // full → drop
        wfRing[wfHead & WFR_MASK] = s[i];
        wfHead++;
    }
}

// Internal DAC via I2S-DMA. Both DAC channels enabled (GPIO25 + GPIO26) fed the
// same mono sample, so the jack works wired to either — we spec GPIO25 (Tip).
static void dacInstall() {
    if (dacInstalled) return;
    i2s_config_t cfg = {};
    cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_DAC_BUILT_IN);
    cfg.sample_rate = REC_AUDIO_RATE * WF_DAC_UPSAMPLE;   // honored rate (see #5938)
    cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
    cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
    cfg.communication_format = I2S_COMM_FORMAT_STAND_MSB;
    cfg.intr_alloc_flags = 0;
    cfg.dma_buf_count = 8;
    cfg.dma_buf_len = 256;
    cfg.use_apll = false;
    if (i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL) != ESP_OK) {
        Serial.println("[REC] DAC i2s install failed");
        return;
    }
    i2s_set_dac_mode(I2S_DAC_CHANNEL_BOTH_EN);
    i2s_zero_dma_buffer(I2S_NUM_0);
    dacInstalled = true;
}
static void dacUninstall() {
    if (!dacInstalled) return;
    i2s_zero_dma_buffer(I2S_NUM_0);
    i2s_driver_uninstall(I2S_NUM_0);
    dacInstalled = false;
}

static void recAudioBegin(uint8_t slot) {
    char path[24];
    snprintf(path, sizeof(path), "/rec_%d.audio", slot);
    recAudioFile  = SPIFFS.open(path, "w");
    recAudioOpen  = (bool)recAudioFile;
    recAudioBytes = 0;
    wfRingReset();
    if (!recAudioOpen) Serial.println("[REC] audio file open failed");
}
static void recAudioDrain() {                                // loop (consumer)
    if (!recAudioOpen) return;
    uint8_t chunk[256];
    while ((wfHead - wfTail) > 0 && recAudioBytes < REC_AUDIO_MAX) {
        int g = 0;
        while ((wfHead - wfTail) > 0 && g < (int)sizeof(chunk)
               && recAudioBytes + g < REC_AUDIO_MAX) {
            chunk[g++] = wfRing[wfTail & WFR_MASK];
            wfTail++;
        }
        if (g > 0) { recAudioFile.write(chunk, g); recAudioBytes += g; }
    }
}
static void recAudioEnd(uint8_t slot) {
    if (!recAudioOpen) return;
    recAudioDrain();
    recAudioFile.flush();
    recAudioFile.close();
    recAudioOpen = false;
    recSlotHasAudio[slot] = (recAudioBytes > 0);
    Serial.printf("[REC] audio slot %u: %lu bytes (%.1fs)\n",
                  slot, (unsigned long)recAudioBytes, recAudioBytes / (float)REC_AUDIO_RATE);
}

static void playbackAudioBegin(uint8_t slot) {
    char path[24];
    snprintf(path, sizeof(path), "/rec_%d.audio", slot);
    playAudioOpen = false;
    if (!SPIFFS.exists(path)) return;                        // legacy slot → envelope
    playAudioFile = SPIFFS.open(path, "r");
    if (!playAudioFile || playAudioFile.size() == 0) return;
    playAudioOpen = true;
    dacInstall();
}
static void playbackAudioEnd() {
    if (playAudioOpen) { playAudioFile.close(); playAudioOpen = false; }
    dacUninstall();
}

// Speaker volume, applied ONLY at the DAC fill below — the fxEnergy/fxOnset
// re-derivation reads the same samples at recorded level, so LED reactivity
// does not change with speaker volume. Each halving costs one of the DAC's
// 8 bits (quantization hiss under speech); below ~0.06 use a hardware
// divider on the jack instead.
static float audioOutVol = 0.12f;

// Feed one frame of recorded audio to the DAC and re-derive fxEnergy/fxOnset
// from those same samples. Returns false when the slot has no waveform, so the
// caller falls back to the legacy companded-envelope bytes.
static bool playbackAudioTick(float dt) {
    if (!playAudioOpen) return false;
    int n = (int)(REC_AUDIO_RATE * dt + 0.5f);
    if (n < 1)   n = 1;
    if (n > 512) n = 512;

    static uint8_t sbuf[512];
    int got = playAudioFile.read(sbuf, n);
    if (got <= 0) { playAudioFile.seek(0); got = playAudioFile.read(sbuf, n); }  // loop
    if (got <= 0) return false;

    // → DAC: 8-bit sample in the high byte of a 16-bit word, duplicated L/R,
    // each source sample repeated WF_DAC_UPSAMPLE× (zero-order hold) so the
    // 24 kHz DAC clock reproduces 8 kHz content at correct pitch. Chunked so
    // dbuf stays 1024 words regardless of upsample (no extra static DRAM).
    if (dacInstalled) {
        static uint16_t dbuf[1024];
        const int SRC_CHUNK = 1024 / (2 * WF_DAC_UPSAMPLE);   // source samples/chunk
        int off = 0;
        while (off < got) {
            int c = (got - off) > SRC_CHUNK ? SRC_CHUNK : (got - off);
            int w = 0;
            for (int i = 0; i < c; i++) {
                int32_t ac = (int32_t)sbuf[off + i] - 128;   // center on midline
                uint16_t v = (uint16_t)(uint8_t)(128 + (int32_t)(ac * audioOutVol)) << 8;
                for (int u = 0; u < WF_DAC_UPSAMPLE; u++) { dbuf[w++] = v; dbuf[w++] = v; }
            }
            size_t wrote = 0;
            i2s_write(I2S_NUM_0, dbuf, (size_t)w * sizeof(uint16_t), &wrote, portMAX_DELAY);
            off += c;
        }
    }

    // → re-derive RMS in the 24-bit domain: ac ≈ (byte-128)<<QUANT_SHIFT.
    // Sub-window the chunk so rms_max captures the loudest slice (mirrors the
    // sender's per-window max), rms_mean is the whole-chunk RMS.
    const int SUB = 4;
    double total = 0.0;
    float  subMax = 0.0f;
    int per = got / SUB; if (per < 1) per = got;
    int idx = 0;
    for (int w = 0; w < SUB && idx < got; w++) {
        double ss = 0.0; int cnt = 0;
        int end = (w == SUB - 1) ? got : (idx + per);
        if (end > got) end = got;
        for (; idx < end; idx++) {
            double a = (double)(((int32_t)sbuf[idx] - 128) << WF_QUANT_SHIFT);
            ss += a * a; total += a * a; cnt++;
        }
        if (cnt > 0) { float r = sqrtf((float)(ss / cnt)); if (r > subMax) subMax = r; }
    }
    float rmsMean = sqrtf((float)(total / got));
    float rmsMax  = fmaxf(subMax, rmsMean);
    audioDeriveFeaturesRms(rmsMean, rmsMax, dt);
    return true;
}

static void recStartRecording() {
    if (!recAllocBuffer()) return;
    recFrameCount = 0;
    recFrameAcc   = 0.0f;
    recAudioBegin(recSlot);
    recState = REC_RECORDING;
    Serial.printf("[REC] recording to slot %u...\n", recSlot);
}

static void recStopRecording() {
    recState = REC_IDLE;
    recAudioEnd(recSlot);          // close /rec_N.audio (even if no frames)
    if (recFrameCount == 0) return;
    char path[20];
    snprintf(path, sizeof(path), "/rec_%d.bin", recSlot);
    File f = SPIFFS.open(path, "w");
    if (!f) { Serial.println("[REC] write failed"); return; }
    RecFileHeader hdr;
    memcpy(hdr.magic, REC_MAGIC, 4);
    hdr.version   = REC_FILE_VERSION;
    hdr.frameSize = (uint8_t)sizeof(RecFrame);
    hdr.reserved  = 0;
    f.write((uint8_t*)&hdr, sizeof(hdr));
    f.write((uint8_t*)recBuffer, recFrameCount * sizeof(RecFrame));
    f.close();
    recSlotHasData[recSlot] = true;
    Serial.printf("[REC] saved slot %u: %lu frames (%.1fs)\n",
                  recSlot, recFrameCount, recFrameCount / (float)REC_FPS);
}

// Defined with the audio globals below; reseeds the adaptive floor/ceiling/
// onset state so playback re-derivation converges from a clean baseline rather
// than inheriting whatever the live audio left behind.
static void audioResetAdaptiveState();

// Reads one slot's frames into recBuffer, decoding the on-disk format.
// Returns the frame count loaded (0 = nothing usable). Legacy headerless
// 6-byte files load with rms bytes zeroed (silent audio on replay).
static uint32_t recLoadSlot(uint8_t slot) {
    char path[20];
    snprintf(path, sizeof(path), "/rec_%d.bin", slot);
    File f = SPIFFS.open(path, "r");
    if (!f) return 0;
    size_t fileSize = f.size();

    // Peek the header to decide versioned vs. legacy.
    RecFileHeader hdr;
    bool versioned = false;
    uint8_t frameSize = REC_LEGACY_FRAME_SIZE;
    if (fileSize >= sizeof(hdr)) {
        f.read((uint8_t*)&hdr, sizeof(hdr));
        if (memcmp(hdr.magic, REC_MAGIC, 4) == 0 && hdr.frameSize > 0
            && hdr.frameSize <= (uint8_t)sizeof(RecFrame)) {
            versioned = true;
            frameSize = hdr.frameSize;
        }
    }

    uint32_t count = 0;
    if (versioned) {
        // Frames follow the header; on-disk frameSize may be <= our struct
        // (older version with fewer fields). Read each frame into a zeroed
        // RecFrame so any missing trailing fields (e.g. rms bytes) default 0.
        uint32_t avail = (fileSize - sizeof(hdr)) / frameSize;
        if (avail > REC_MAX_FRAMES) avail = REC_MAX_FRAMES;
        for (uint32_t i = 0; i < avail; i++) {
            RecFrame rf = {};
            if (f.read((uint8_t*)&rf, frameSize) != frameSize) break;
            recBuffer[i] = rf;
            count++;
        }
    } else {
        // Legacy headerless file: tightly-packed 6-byte frames, no audio.
        f.seek(0);
        uint32_t avail = fileSize / REC_LEGACY_FRAME_SIZE;
        if (avail > REC_MAX_FRAMES) avail = REC_MAX_FRAMES;
        for (uint32_t i = 0; i < avail; i++) {
            RecFrame rf = {};
            if (f.read((uint8_t*)&rf, REC_LEGACY_FRAME_SIZE) != REC_LEGACY_FRAME_SIZE) break;
            recBuffer[i] = rf;   // rms_mean/rms_max remain 0 (silent on replay)
            count++;
        }
    }
    f.close();
    return count;
}

static void recStartPlayback() {
    if (!recSlotHasData[recSlot]) {
        Serial.printf("[REC] slot %u empty\n", recSlot);
        return;
    }
    if (!recAllocBuffer()) return;
    recFrameCount = recLoadSlot(recSlot);
    if (recFrameCount == 0) {
        Serial.printf("[REC] slot %u unreadable\n", recSlot);
        return;
    }
    recPlayIdx = 0;
    recPlayClock = 0.0f;
    // Reseed audio adaptive state so the re-derived energy/onset converges
    // cleanly from this recording rather than from prior live audio.
    audioResetAdaptiveState();
    playbackAudioBegin(recSlot);   // open /rec_N.audio + DAC if this slot has one
    recState = REC_PLAYING;
    Serial.printf("[REC] playing slot %u: %lu frames\n", recSlot, recFrameCount);
}

static void recStopPlayback() {
    recState = REC_IDLE;
    playbackAudioEnd();            // close audio + free the DAC/I2S peripheral
    Serial.println("[REC] playback stopped");
}

#define GATE_ON_THRESH  0.7f
#define GATE_OFF_THRESH 1.2f

#define SENSOR_HZ       25.0f

// Tilt mapping (used by ported sparkle effect)
#define DEADZONE_DEG    10.0f
#define MAX_ANGLE_DEG   180.0f

// ── Effect drive parameters ──────────────────────────────────────
// BS-26 live → knobs control. No signal → serial PARAMS control.
static float globalBrightness = 1.0f;   // serial-controlled, 0.0–1.0
static float speedScale       = 1.0f;   // serial-controlled, 0.2–3.0
static float renderBrightness = 1.0f;   // ceiling: max output scale
static float renderFloor      = 0.0f;   // reserved — was proportional floor (removed: killed animation dynamics)
static float gDt = 1.0f / 60.0f;        // current frame dt, set each loop iteration

// ── BS-26 ESP-NOW config ─────────────────────────────────────────
#define FIXED_CHANNEL        1
#define HEARTBEAT_TIMEOUT_MS 6000

// ── BS-26 state ──────────────────────────────────────────────────
struct Bs26State {
    bool     dc;
    bool     ac;
    uint8_t  hundreds;
    uint8_t  tens;
    uint8_t  ones;
    uint16_t decade;     // 0–1199
    uint8_t  decouter;   // 0–11, outer ring
    uint8_t  decmid;     // 0–9, middle ring
    uint8_t  decinner;   // 0–9, inner ring
    uint8_t  ua_dac;     // 0–255
    uint8_t  ua_max;
    uint8_t  v_dac;
    uint8_t  v_max;
    uint32_t seq;
};

static volatile Bs26State bs26 = {};
static volatile uint32_t  bs26LastMs = 0;
static volatile uint32_t  bs26PktCount = 0;
static volatile bool      bs26Updated = false;
// Guards the multi-field bs26 write (onReceive task) against the bulk read
// (loop). Without it a torn read can momentarily mix fields and spike brightness.
static portMUX_TYPE bs26Mux = portMUX_INITIALIZER_UNLOCKED;

// ── Sensor layer (v1 accel / gyro / audio packets over ESP-NOW) ──
// Coexists with BS-26 JSON; disambiguated by packet length in onReceive.
// Raw packet fields are latched in the callback; loop() decodes + normalizes.
#define SENSOR_TIMEOUT_MS 3000

// Companding full-scale (matches v1 packet encoders).
#define V1_AMAG_FS  57000.0f
#define V1_GMAG_FS  57000.0f
#define V1_RMS_FS   200000.0f   // must match sender_v2 RMS_FS

// Production sender locks ±4g accel / ±1000 dps gyro.
#define V1_ACCEL_RANGE_G    4.0f
#define V1_GYRO_RANGE_DPS   1000.0f
#define V1_COUNTS_PER_G     (32768.0f / V1_ACCEL_RANGE_G)
#define V1_DPS_PER_LSB      (V1_GYRO_RANGE_DPS / 128.0f)   // int8 mean → dps

static volatile TelemetryPacketV1 accelPkt = {};
static volatile uint32_t accelLastMs = 0;
static volatile uint32_t accelPktCount = 0;

static volatile GyroPacketV1 gyroPkt = {};
static volatile uint32_t gyroLastMs = 0;
static volatile uint32_t gyroPktCount = 0;

static volatile AudioPacketV1 audioPkt = {};
static volatile uint32_t audioLastMs = 0;
static volatile uint32_t audioPktCount = 0;

// sqrt-companded uint8 → linear value: val = (byte/255)² * FS
static inline float decodeCompand(uint8_t b, float fs) {
    float t = (float)b / 255.0f;
    return t * t * fs;
}

// ── PRNG ─────────────────────────────────────────────────────────
static uint32_t prngState;

static inline uint32_t xorshift32() {
    prngState ^= prngState << 13;
    prngState ^= prngState >> 17;
    prngState ^= prngState << 5;
    return prngState;
}

static inline float randFloat() {
    return (float)(xorshift32() & 0xFFFFFF) / 16777216.0f;
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// ── HSV ↔ RGB (h 0..360, s/v 0..1, rgb 0..255) ─────────────────
static void hsvToRgb(float h, float s, float v, float &r, float &g, float &b) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r1, g1, b1;
    if      (h < 60)  { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
    else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
    else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
    else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
    else              { r1 = c; g1 = 0; b1 = x; }
    r = (r1 + m) * 255.0f;
    g = (g1 + m) * 255.0f;
    b = (b1 + m) * 255.0f;
}

static void rgbToHsv(float r, float g, float b, float &h, float &s, float &v) {
    r /= 255.0f; g /= 255.0f; b /= 255.0f;
    float mx = fmaxf(fmaxf(r, g), b);
    float mn = fminf(fminf(r, g), b);
    float d = mx - mn;
    v = mx;
    s = (mx <= 0.0f) ? 0.0f : d / mx;
    if (d <= 0.0f)        h = 0.0f;
    else if (mx == r)     h = 60.0f * fmodf((g - b) / d, 6.0f);
    else if (mx == g)     h = 60.0f * ((b - r) / d + 2.0f);
    else                  h = 60.0f * ((r - g) / d + 4.0f);
    if (h < 0.0f) h += 360.0f;
}

// ── Bloom hue endpoints (runtime, rotated by palette index) ─────
#define BLOOM_HUE_A_R_BASE    0.0f
#define BLOOM_HUE_A_G_BASE  180.0f
#define BLOOM_HUE_A_B_BASE  120.0f
#define BLOOM_HUE_B_R_BASE  140.0f
#define BLOOM_HUE_B_G_BASE   20.0f
#define BLOOM_HUE_B_B_BASE  255.0f

static float bloomHueA_R = BLOOM_HUE_A_R_BASE;
static float bloomHueA_G = BLOOM_HUE_A_G_BASE;
static float bloomHueA_B = BLOOM_HUE_A_B_BASE;
static float bloomHueB_R = BLOOM_HUE_B_R_BASE;
static float bloomHueB_G = BLOOM_HUE_B_G_BASE;
static float bloomHueB_B = BLOOM_HUE_B_B_BASE;

static uint8_t paletteHueIdx = 0;   // 0–9, from decmid
static uint8_t paletteSatIdx = 5;   // 0–9, from decinner; 5 = neutral (originals)

// Saturation knob curve: 5 leaves the effect's original saturation untouched.
// 0–4 boosts toward full saturation, linear in knob position, weighted by
// existing saturation with an ease-out (s·(2−s)) so mid-vivid colors push
// hard while neutral pixels (s≈0, hue meaningless) stay neutral. 6–9
// desaturates on a quadratic falloff so 6–7 stay colorful and only 8–9
// read as washed → white.
static inline float satKnob(float s, uint8_t satIdx) {
    if (satIdx < 5) {
        float t = (5.0f - (float)satIdx) / 5.0f;
        float w = s * (2.0f - s);
        return s + (1.0f - s) * t * w;
    }
    float u = (float)(satIdx - 5) / 4.0f;
    return s * (1.0f - u * u);
}

// Rotate base endpoint hues by hueIdx * 36° and re-saturate by satIdx.
static void applyPalette(uint8_t hueIdx, uint8_t satIdx) {
    float rot = (float)hueIdx * 36.0f;
    float h, s, v, r, g, b;
    rgbToHsv(BLOOM_HUE_A_R_BASE, BLOOM_HUE_A_G_BASE, BLOOM_HUE_A_B_BASE, h, s, v);
    hsvToRgb(fmodf(h + rot, 360.0f), satKnob(s, satIdx), v, r, g, b);
    bloomHueA_R = r; bloomHueA_G = g; bloomHueA_B = b;
    rgbToHsv(BLOOM_HUE_B_R_BASE, BLOOM_HUE_B_G_BASE, BLOOM_HUE_B_B_BASE, h, s, v);
    hsvToRgb(fmodf(h + rot, 360.0f), satKnob(s, satIdx), v, r, g, b);
    bloomHueB_R = r; bloomHueB_G = g; bloomHueB_B = b;
}

// ── ESP-NOW receive (BS-26 JSON state packets) ──────────────────
static void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len < 2 || len > 250) return;

    // Fixed-size binary sensor packets take priority over JSON (disambiguated
    // by exact length). Latch raw bytes; loop() decodes and normalizes.
    if (len == (int)sizeof(TelemetryPacketV1)) {
        memcpy((void*)&accelPkt, data, sizeof(TelemetryPacketV1));
        accelLastMs = millis();
        accelPktCount++;
        return;
    }
    if (len == (int)sizeof(GyroPacketV1)) {
        memcpy((void*)&gyroPkt, data, sizeof(GyroPacketV1));
        gyroLastMs = millis();
        gyroPktCount++;
        return;
    }
    if (len == (int)sizeof(AudioPacketV1)) {
        memcpy((void*)&audioPkt, data, sizeof(AudioPacketV1));
        audioLastMs = millis();
        audioPktCount++;
        return;
    }
    if (len == (int)sizeof(AudioStreamPacketV1)) {
        // Lo-fi waveform from sender_v3 — only flowing while a slot records.
        if (recState == REC_RECORDING) {
            const AudioStreamPacketV1 *wp = (const AudioStreamPacketV1 *)data;
            uint8_t n = wp->n;
            if (n > AUDIO_STREAM_SAMPLES) n = AUDIO_STREAM_SAMPLES;
            wfRingPush(wp->samples, n);
        }
        return;
    }

    StaticJsonDocument<768> doc;
    DeserializationError err = deserializeJson(doc, (const char*)data, len);
    if (err) return;

    if (!doc.containsKey("s") && !doc.containsKey("seq")) return;

    // Build the full state in a scratch copy, then publish it to bs26 in one
    // critical section so loop never reads a half-updated struct.
    Bs26State next = {};
    // ac/dc arrive as integers ("ac":1) over ESP-NOW, not JSON booleans. The
    // `| false` default-operator gates on is<bool>(), which is FALSE for an
    // integer, so it always returned the default — switches never registered.
    // as<bool>() converts the integer (1→true) instead of type-gating it.
    next.dc       = doc["dc"].as<bool>();
    next.ac       = doc["ac"].as<bool>();
    next.hundreds = doc["h"]  | (doc["hundreds"] | 0);
    next.tens     = doc["t"]  | (doc["tens"] | 0);
    next.ones     = doc["o"]  | (doc["ones"] | 0);
    next.decade   = doc["dec"]| (doc["decade"] | 0);
    next.decouter = doc["do"] | (doc["decouter"] | 0);
    next.decmid   = doc["dm"] | (doc["decmid"] | 0);
    next.decinner = doc["di"] | (doc["decinner"] | 0);
    next.ua_dac   = doc["ua"] | 0;
    next.ua_max   = 0;
    next.v_dac    = doc["v"]  | 0;
    next.v_max    = 0;
    next.seq      = doc["s"]  | (doc["seq"] | 0);

    portENTER_CRITICAL(&bs26Mux);
    memcpy((void*)&bs26, &next, sizeof(bs26));
    portEXIT_CRITICAL(&bs26Mux);

    bs26LastMs = millis();
    bs26PktCount++;
    bs26Updated = true;
}

// ── LED driver: 6 strips via RMT ─────────────────────────────────
// Data pins default to the tree-of-record wiring. Override per-board via
// build flags — e.g. biolum-a wires strip1 to GPIO 16 instead of 15.
#ifndef STRIP0_PIN
#define STRIP0_PIN 4
#endif
#ifndef STRIP1_PIN
#define STRIP1_PIN 15
#endif
#ifndef STRIP2_PIN
#define STRIP2_PIN 17
#endif
#ifndef STRIP3_PIN
#define STRIP3_PIN 5
#endif
#ifndef STRIP4_PIN
#define STRIP4_PIN 18
#endif
#ifndef STRIP5_PIN
#define STRIP5_PIN 19
#endif
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt0Ws2812xMethod> strip0(LEDS_PER_STRIP, STRIP0_PIN);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt1Ws2812xMethod> strip1(LEDS_PER_STRIP, STRIP1_PIN);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt2Ws2812xMethod> strip2(LEDS_PER_STRIP, STRIP2_PIN);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt3Ws2812xMethod> strip3(LEDS_PER_STRIP, STRIP3_PIN);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt4Ws2812xMethod> strip4(LEDS_PER_STRIP, STRIP4_PIN);
static NeoPixelBus<NeoRgbFeature, NeoEsp32Rmt5Ws2812xMethod> strip5(LEDS_PER_STRIP, STRIP5_PIN);

static inline void setPixel(uint8_t s, uint16_t i, uint8_t r, uint8_t g, uint8_t b) {
    RgbColor c(r, g, b);
    switch (s) {
        case 0: strip0.SetPixelColor(i, c); break;
        case 1: strip1.SetPixelColor(i, c); break;
        case 2: strip2.SetPixelColor(i, c); break;
        case 3: strip3.SetPixelColor(i, c); break;
        case 4: strip4.SetPixelColor(i, c); break;
        case 5: strip5.SetPixelColor(i, c); break;
    }
}

static void showAll() {
    strip0.Show(); strip1.Show(); strip2.Show();
    strip3.Show(); strip4.Show(); strip5.Show();
}

// ── Stick topology ──────────────────────────────────────────────
// Physical mapping (from topo_test): each "stick" is a contiguous LED range
// on a strip. start/end are inclusive LED indices. Strips 1 and 3 have no
// sticks. Hand-measured against the static block test pattern.
struct Stick { uint8_t strip; uint8_t start; uint8_t end; };
static const Stick STICKS[] = {
    { 0, 18,  22 }, { 0, 27,  49 }, { 0, 54,  99 },
    { 2, 14,  50 },
    { 4, 18,  38 }, { 4, 47,  99 },
    { 5, 14,  30 }, { 5, 45,  74 }, { 5, 83,  99 },
};
static const uint8_t STICK_COUNT = sizeof(STICKS) / sizeof(STICKS[0]);

// ── Generic output stage (shared by ported effects) ─────────────
// Ported bulb-fleet/biolum effects write 0–255-range float color here;
// this scales by renderBrightness and truncates to 8-bit. No dithering.

// ── Hysteretic floor: anti-flicker policy for sub-LSB channels ───
// Each channel per pixel has a timer (seconds, float). Positive = time spent
// below threshold (trending toward dormant). Negative = time spent above
// (trending toward active). State transitions:
//   ACTIVE  → DORMANT: timer reaches +FLOOR_DEACTIVATE_S (channel goes dark)
//   DORMANT → ACTIVE:  timer reaches -FLOOR_REACTIVATE_S (channel re-enables)
// While active and sub-threshold: output floored at 1/255 (steady, no flicker).
// While dormant: output true zero regardless of target.
#define FLOOR_DEACTIVATE_S  1.0f
#define FLOOR_REACTIVATE_S  0.5f
#define FLOOR_THRESHOLD     256    // target16 below this = sub-LSB danger zone

static float fxFloorTimer[NUM_STRIPS][LEDS_PER_STRIP][1];

static void resetFxDither() {
    memset(fxFloorTimer, 0, sizeof(fxFloorTimer));
}

static inline uint16_t applyFloorPolicy(uint16_t target16, float &timer, float dt) {
    if (target16 < FLOOR_THRESHOLD) {
        timer += dt;
        if (timer >= FLOOR_DEACTIVATE_S) {
            timer = FLOOR_DEACTIVATE_S;
            return 0;  // dormant: true zero
        }
        return FLOOR_THRESHOLD;  // active but sub-LSB: floor at 1/255
    } else {
        timer -= dt;
        if (timer <= -FLOOR_REACTIVATE_S) {
            timer = 0.0f;  // fully reactivated
            return target16;
        }
        if (timer > 0.0f) {
            return 0;  // was dormant, waiting to reactivate
        }
        return target16;  // active, above threshold
    }
}

// Global palette transform: rotate hue by paletteHueIdx×36° and re-saturate
// via satKnob() (5 = original, <5 boost, >5 desaturate), matching
// applyPalette()'s rule. This is the shared path for every effect that routes
// through setPixelScaled, so the decmid/decinner knobs affect all of them.
// Bloom does NOT come through here (it uses setPixel directly and rotates its
// own base colors), so there is no double-rotation. No-op fast path at the
// neutral knob position.
static inline void applyGlobalPalette(float &r, float &g, float &b) {
    if (paletteHueIdx == 0 && paletteSatIdx == 5) return;
    float h, sv, v;
    rgbToHsv(r, g, b, h, sv, v);
    h = fmodf(h + (float)paletteHueIdx * 36.0f, 360.0f);
    sv = satKnob(sv, paletteSatIdx);
    hsvToRgb(h, sv, v, r, g, b);
}

// fr/fg/fb in 0–255 linear range (already gamma-shaped by the effect).
static inline void setPixelScaled(uint8_t s, uint16_t i, float fr, float fg, float fb) {
    applyGlobalPalette(fr, fg, fb);
    if (fr > 0.0f || fg > 0.0f || fb > 0.0f) {
        fr *= renderBrightness;
        fg *= renderBrightness;
        fb *= renderBrightness;
    }
    uint16_t t16R = (uint16_t)fminf(fr * 256.0f, 65535.0f);
    uint16_t t16G = (uint16_t)fminf(fg * 256.0f, 65535.0f);
    uint16_t t16B = (uint16_t)fminf(fb * 256.0f, 65535.0f);
    uint16_t maxT16 = t16R > t16G ? (t16R > t16B ? t16R : t16B)
                                   : (t16G > t16B ? t16G : t16B);
    uint16_t floorResult = applyFloorPolicy(maxT16, fxFloorTimer[s][i][0], gDt);
    if (floorResult == 0 && maxT16 < FLOOR_THRESHOLD) {
        t16R = t16G = t16B = 0;
    }
    uint8_t r8 = t16R >> 8;
    uint8_t g8 = t16G >> 8;
    uint8_t b8 = t16B >> 8;
    setPixel(s, i, r8, g8, b8);
}

// Per-strip OKLCH hue offsets (~60° apart on 256-step wheel) so the 6
// strips span a rainbow chord. From bulb-fleet.
static const uint8_t STRIP_HUE_OFFSET[NUM_STRIPS] = { 0, 42, 85, 128, 170, 213 };

// ── Effect selection (BS-26 decouter ring, 0–11) ────────────────
// Knob order groups by what drives the effect: audio-reactive first (0–4,
// quietest response to busiest), then ambient effects that ignore the
// sensors entirely (5–8, organic → cosmic → pure color).
// Positions ≥ FX_COUNT clamp to the last effect.
enum EffectId : uint8_t {
    // audio-reactive: field effects
    FX_AMBIENT_BLOOM = 0,   // slow energy swell (audioEnergy5s contrast)
    FX_FIRE,                // energy = flame height, onsets = flicker
    FX_HEAT_DIFFUSION,      // energy injects heat at rotating point; PDE spreads it
    FX_WORLEY,              // onsets kick Voronoi seeds; boundaries flare
    // audio-reactive: particle effects
    FX_GRAVITY_PARTICLE,    // speech envelope springs particles outward
    FX_SPARKLE_SYLLABLE,    // discrete onsets spawn sparks (+ tilt hue)
    FX_CREATURES,           // speech-energy excitement (was accel shake)
    // ambient: particle effects
    FX_POLYCULE,            // rainbow-tailed particles bounce + collide
    FX_LEAF_WIND,
    // ambient: passive
    FX_LIGHT_THROUGH,
    FX_NEBULA,
    FX_RAINBOW,
    FX_COUNT                // = 12: fills the decouter ring exactly
};

static uint8_t currentEffect = FX_AMBIENT_BLOOM;
static uint8_t prevEffect    = 0xFF;   // forces reset on first frame

// ── Shared per-effect buffers (DRAM is tight) ────────────────────
// Effects are mutually exclusive and fully re-initialized on switch
// (resetEffect), so large per-pixel state can share one allocation.
// Add a union member per effect that needs a field; never read across
// effects, and make sure the effect's reset writes every cell it reads.
static union {
    float neb[NUM_STRIPS][LEDS_PER_STRIP];   // nebula: orb trail decay
    float heat[NUM_STRIPS][LEDS_PER_STRIP];  // heat diffusion: temperature
} fxField;

// Per-strip RGB accumulation scratch, valid within one render call only
// (gravity, polycule).
static float fxAccR[LEDS_PER_STRIP], fxAccG[LEDS_PER_STRIP], fxAccB[LEDS_PER_STRIP];

// ── Audio/motion feature globals ────────────────────────────────
// The bulb-fleet effects (gravity, sparkle, fire, quiet bloom) consume
// audio/motion features. processSensors() drives these every frame from the
// v1 accel/gyro/audio packets. On sensor timeout they fall back to rest
// (energy/onset = 0, no tilt, accel flat) so each effect's idle/ambient
// branch renders unchanged.
struct SensorStub {
    int16_t ax, ay, az;   // accel counts (16384 = 1g); flat = (0,0,16384)
    int16_t gx, gy, gz;
};
static SensorStub sensorStub = { 0, 0, 16384, 0, 0, 0 };
static float fxEnergy   = 0.0f;   // 0–1 audio/motion energy
static float fxOnset    = 0.0f;   // 0–1 transient onset
static float fxTiltBlend = 0.0f;  // 0–1 tilt engagement
static float fxAngleDeg = 0.0f;   // tilt angle for hue mapping

// "Energy in the system": leaky integral of fxEnergy over ~tau seconds. Builds
// while audio is present, bleeds off over ~tau when it stops. Drives the bloom
// contrast stretch. energyForce >= 0 overrides it for live tuning without audio.
static float audioEnergy5s = 0.0f;
static float energy5sTau   = 3.0f;   // integration/decay window (s)
static float energyForce   = -1.0f;  // -1 = use real audio; 0–1 = forced override

// ── Sensor → feature processing ──────────────────────────────────
// Decodes the v1 packets and drives the effect globals. Audio uses an
// adaptive floor + log scaling (ported from bulb-fleet computeEnergy/Onset,
// retuned for the decoded RMS scale). On timeout each layer falls back to its
// at-rest stub so the effects render their idle/ambient branch unchanged.
// Hard energy floor: below this, mean RMS reads as ambient/no-intent and
// fxEnergy is zeroed. Measured @ FS=200000 (this mic): silent room ~2k,
// room music ~16-37k, distant speech ~13-37k, direct speech peaks 92-133k.
// 20k sits above music/distant talk so only deliberate close speech drives
// energy. The adaptive floor floats above this when ambient is louder.
#define SNS_RMS_FLOOR_MIN  20000.0f
// Ceiling seed + leak floor. The ceiling used to boot at the 20k floor min —
// below the 28k effective floor — which clamped dbRange to its 1 dB minimum
// and mapped the FIRST sound after boot/playback-start to fxEnergy = 1.0
// regardless of loudness (gravity slammed outward at first speech). Seed at
// the measured direct-close-speech scale (peaks 92-133k, max 141k @
// FS=200000) so the first utterance lands ~0.7-1.0 honestly; louder peaks
// still raise the ceiling above this.
#define SNS_RMS_CEIL_MIN   130000.0f
// Onset presence floor — gates onset on the absolute window MAX (not mean), so a
// sharp transient over a quiet bed still counts (clap-to-interact). Set just
// above the measured silent-room noise ceiling (~2.6k max) so mic-noise jitter
// can't manufacture phantom onsets, but far below any real sound (≥17k) so
// genuine soft transients still register. Distinct from the 20k energy floor:
// that's the opt-in/taste knob; this is a noise-derived phantom kill.
#define SNS_ONSET_FLOOR    6000.0f
#define SNS_FLOOR_HEADROOM 1.4f
#define SNS_FLOOR_LEAK     0.005f
#define SNS_FLOOR_SNAP_EPS 0.05f
#define SNS_FLOOR_SOFT_SIG 0.6f

static float snsAdaptiveFloor = 0.0f;
static float snsRmsCeiling    = SNS_RMS_CEIL_MIN;
static int   snsBelowFloorCnt = 0;

// Energy-flux onset detector state + feel knobs (tunable live via !P).
// Onset = rmsMean rising sharply above its recent baseline — a discrete event
// at each attack, unlike the old self-normalizing max/mean ratio that sat near
// 1.0 through all continuous sound and fired on the noise floor.
static float onsetFast    = 0.0f;   // ~30 ms energy envelope
static float onsetSlow    = 0.0f;   // ~300 ms baseline
static float onsetLock    = 0.0f;   // refractory countdown (s)
static float onsetRefrac  = 0.120f; // refractory lockout: max ~1/this fires/s
static float onsetRiseFrac= 0.5f;   // fire when (fast-slow) > riseFrac*slow (floored at SNS_ONSET_FLOOR)

static float snsUpdateFloor(float rms, float dt) {
    if (snsAdaptiveFloor < 1.0f) {
        snsAdaptiveFloor = fmaxf(rms, SNS_RMS_FLOOR_MIN);
        return snsAdaptiveFloor;
    }
    if (rms < snsAdaptiveFloor * (1.0f + SNS_FLOOR_SNAP_EPS)) {
        if (++snsBelowFloorCnt >= 3) {
            float target = fmaxf(rms, SNS_RMS_FLOOR_MIN);
            float snapAlpha = fminf(1.0f, dt / 0.11f);
            snsAdaptiveFloor += snapAlpha * (target - snsAdaptiveFloor);
        }
    } else {
        snsBelowFloorCnt = 0;
        float ratio = rms / fmaxf(snsAdaptiveFloor, 1.0f);
        float d = (ratio - 1.0f) / SNS_FLOOR_SOFT_SIG;
        snsAdaptiveFloor *= (1.0f + SNS_FLOOR_LEAK * dt * expf(-(d * d)));
    }
    snsAdaptiveFloor = fmaxf(snsAdaptiveFloor, SNS_RMS_FLOOR_MIN);
    return snsAdaptiveFloor;
}

// Reseed the adaptive floor/ceiling/onset state to declaration defaults. Called
// at playback start so re-derived energy/onset converges cleanly instead of
// inheriting the prior live-audio state.
static void audioResetAdaptiveState() {
    snsAdaptiveFloor = 0.0f;
    snsRmsCeiling    = SNS_RMS_CEIL_MIN;
    snsBelowFloorCnt = 0;
    onsetFast = 0.0f;
    onsetSlow = 0.0f;
    onsetLock = 0.0f;
}

// Stage-2 → stage-3 audio derivation: companded RMS bytes → fxEnergy/fxOnset
// via the adaptive floor + log scaling against a leaky ceiling, plus onset from
// the window max/mean gap. Single source of truth for both live packets and
// recorded-frame playback, so silent replay reproduces the captured drive.
// Byte-envelope entry point: decode the companded RMS bytes (live AudioPacketV1
// and legacy recorded frames) and run the shared float core below.
static void audioDeriveFeaturesRms(float rmsMean, float rmsMax, float dt);
static void audioDeriveFeatures(uint8_t rmsMeanByte, uint8_t rmsMaxByte, float dt) {
    audioDeriveFeaturesRms(decodeCompand(rmsMeanByte, V1_RMS_FS),
                           decodeCompand(rmsMaxByte,  V1_RMS_FS), dt);
}

// Core: RMS floats (24-bit domain) → fxEnergy/fxOnset. Single source of truth
// for live packets, legacy envelope frames, and waveform-derived playback.
static void audioDeriveFeaturesRms(float rmsMean, float rmsMax, float dt) {
    // Energy: adaptive floor + log scaling against a leaky ceiling.
    snsRmsCeiling = fmaxf(SNS_RMS_CEIL_MIN, snsRmsCeiling * expf(-0.0025f * dt));
    if (rmsMean > snsRmsCeiling) snsRmsCeiling = rmsMean;
    snsUpdateFloor(rmsMean, dt);
    float effFloor = snsAdaptiveFloor * SNS_FLOOR_HEADROOM;
    if (rmsMean < effFloor) {
        fxEnergy = 0.0f;
    } else {
        float db = 20.0f * log10f(rmsMean / effFloor);
        float dbRange = 20.0f * log10f(snsRmsCeiling / effFloor);
        if (dbRange < 1.0f) dbRange = 1.0f;
        fxEnergy = clampf(db / dbRange, 0.0f, 1.0f);
    }

    // Onset: positive flux of the energy envelope. A syllable/word attack is
    // rmsMean RISING above its recent baseline — a discrete event. The old
    // detector used the within-window max/mean gap normalized by a decaying
    // peak, which sat near 1.0 through all continuous sound and fired on the
    // noise floor (over-active sparkle). Two EMAs (fast envelope, slow
    // baseline); fire on the rise crossing an adaptive threshold, then a
    // refractory lockout swallows the attack's sustain so it fires once.
    onsetFast += fminf(1.0f, dt / 0.030f) * (rmsMean - onsetFast);
    onsetSlow += fminf(1.0f, dt / 0.300f) * (rmsMean - onsetSlow);
    float rise   = onsetFast - onsetSlow;
    float thresh = fmaxf(SNS_ONSET_FLOOR, onsetRiseFrac * onsetSlow);  // self-scales to the room
    onsetLock = fmaxf(0.0f, onsetLock - dt);
    if (rise > thresh && onsetLock <= 0.0f) {
        onsetLock = onsetRefrac;
        fxOnset = clampf(rise / fmaxf(onsetSlow, 1.0f), 0.0f, 1.0f);   // strength for sparkle density
    } else {
        fxOnset = 0.0f;                                                // discrete: zero between events
    }
}

static void processSensors(float dt) {
    uint32_t now = millis();

    // ── Accel ────────────────────────────────────────────────────
    // Liveness deltas are signed: the RX callback updates xLastMs
    // asynchronously, so a packet landing after `now` was captured makes
    // (now - xLastMs) wrap to ~2^32 unsigned — a fresh packet would read as
    // timed out, zeroing energy and reseeding the adaptive state for a frame.
    if (accelLastMs > 0 && (int32_t)(now - accelLastMs) < (int32_t)SENSOR_TIMEOUT_MS) {
        TelemetryPacketV1 a;
        memcpy(&a, (const void*)&accelPkt, sizeof(a));

        // Per-axis means are (rawMean >> 8) → gravity/tilt vector in counts.
        // Reconstruct counts at the full-scale used by the stub (16384 = 1g).
        sensorStub.ax = (int16_t)((int)a.ax_mean << 8);
        sensorStub.ay = (int16_t)((int)a.ay_mean << 8);
        sensorStub.az = (int16_t)((int)a.az_mean << 8);

        float axg = (float)sensorStub.ax / 16384.0f;
        float ayg = (float)sensorStub.ay / 16384.0f;
        float azg = (float)sensorStub.az / 16384.0f;

        // Tilt: angle of the gravity vector away from the z (flat) axis.
        float horiz = sqrtf(axg * axg + ayg * ayg);
        fxAngleDeg = atan2f(horiz, fabsf(azg)) * 180.0f / (float)M_PI;
        fxTiltBlend = clampf((fxAngleDeg - DEADZONE_DEG)
                             / (MAX_ANGLE_DEG - DEADZONE_DEG), 0.0f, 1.0f);
    } else {
        sensorStub.ax = 0; sensorStub.ay = 0; sensorStub.az = 16384;
        fxAngleDeg = 0.0f;
        fxTiltBlend = 0.0f;
    }

    // ── Gyro ─────────────────────────────────────────────────────
    if (gyroLastMs > 0 && (int32_t)(now - gyroLastMs) < (int32_t)SENSOR_TIMEOUT_MS) {
        GyroPacketV1 g;
        memcpy(&g, (const void*)&gyroPkt, sizeof(g));
        // gz_mean is the yaw-rate mean (int8 LSB ≈ V1_DPS_PER_LSB dps).
        sensorStub.gz = (int16_t)((float)g.gz_mean * V1_DPS_PER_LSB);
    } else {
        sensorStub.gz = 0;
    }

    // ── Audio ────────────────────────────────────────────────────
    // During playback the recorded frame's audio bytes drive fxEnergy/fxOnset
    // (applied in the playback block below); live audio is bypassed here so the
    // two sources never both update the adaptive state in one frame.
    if (recState == REC_PLAYING) {
        // No-op: see the playback block, which calls audioDeriveFeatures().
    } else if (audioLastMs > 0 && (int32_t)(now - audioLastMs) < (int32_t)SENSOR_TIMEOUT_MS) {
        AudioPacketV1 au;
        memcpy(&au, (const void*)&audioPkt, sizeof(au));
        audioDeriveFeatures(au.rms_mean, au.rms_max, dt);
    } else {
        // Audio packets stopped (sender off / out of range). Zero the outputs and
        // reset the adaptive floor/ceiling/onset-peak so they don't freeze stale
        // through the dropout and resume mid-decay — packets returning re-seed
        // cleanly, same as at playback start.
        fxEnergy = 0.0f;
        fxOnset = 0.0f;
        audioResetAdaptiveState();
    }

    // Slow "energy in the system": leaky integral of fxEnergy. Builds while
    // audio is present, bleeds off over ~energy5sTau when it stops.
    float a5 = fminf(1.0f, dt / energy5sTau);
    audioEnergy5s += a5 * (fxEnergy - audioEnergy5s);
}

// ── Per-strip bloom state ────────────────────────────────────────
struct BloomStrip {
    float breathPhase[LEDS_PER_STRIP];
    float breathPeriod[LEDS_PER_STRIP];
    float breathPeak[LEDS_PER_STRIP];
    float hueT[LEDS_PER_STRIP];
    float hueDrift[LEDS_PER_STRIP];
    float blackoutTimer[LEDS_PER_STRIP];
    float dormancyDur[LEDS_PER_STRIP];
    float dormancyRoll[LEDS_PER_STRIP];
    bool  gateOff[LEDS_PER_STRIP];
    uint16_t ditherR[LEDS_PER_STRIP];
    uint16_t ditherG[LEDS_PER_STRIP];
    uint16_t ditherB[LEDS_PER_STRIP];
    float flash;
    float energyBuffer;
};

static BloomStrip bloom[NUM_STRIPS];

static void resetBloomStrip(BloomStrip &bs) {
    bs.flash = 0.0f;
    bs.energyBuffer = 0.0f;
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        bs.breathPhase[i]  = randFloat();
        bs.breathPeriod[i] = BLOOM_BREATH_MIN_PERIOD
            + randFloat() * (BLOOM_BREATH_MAX_PERIOD - BLOOM_BREATH_MIN_PERIOD);
        bs.breathPeak[i]   = BLOOM_BREATH_MIN_PEAK
            + randFloat() * (BLOOM_BREATH_MAX_PEAK - BLOOM_BREATH_MIN_PEAK);
        bs.hueT[i]         = randFloat();
        float rate = BLOOM_HUE_DRIFT_MIN
            + randFloat() * (BLOOM_HUE_DRIFT_MAX - BLOOM_HUE_DRIFT_MIN);
        bs.hueDrift[i]     = (randFloat() > 0.5f) ? rate : -rate;
        bs.blackoutTimer[i] = 0.0f;
        bs.dormancyDur[i]   = bloomDormancyMin + randFloat() * (bloomDormancyMax - bloomDormancyMin);
        bs.dormancyRoll[i]  = randFloat();
        bs.gateOff[i]      = false;
        bs.ditherR[i] = 0;
        bs.ditherG[i] = 0;
        bs.ditherB[i] = 0;
    }
}

// ── Bloom render (per strip, ambient only — no motion) ──────────
static void renderBloomStrip(uint8_t s, float dt) {
    BloomStrip &bs = bloom[s];

    // No motion processing — ambient breathing only.
    // Flash/energyBuffer stay at zero (no input).

    float drain = bs.energyBuffer * bloomBufferDrain * dt;
    if (bs.energyBuffer < 0.05f) drain += 0.1f * dt;
    drain = fminf(drain, bs.energyBuffer);
    bs.energyBuffer -= drain;
    bs.flash = fminf(1.0f, bs.flash + drain);

    bs.flash *= expf(-bloomFlashDecayRate * dt);
    if (bs.flash < 0.005f) bs.flash = 0.0f;

    float flashLin = bs.flash * bloomBrightnessCap;
    float flashFrac = (bs.flash > 0.1f) ? clampf(bs.flash / 0.3f, 0.0f, 1.0f) : 0.0f;
    float oneMinusFF = 1.0f - flashFrac;

    // Pass 1: breath glow per pixel + field mean (the pivot for the stretch).
    float glow[LEDS_PER_STRIP];
    float glowSum = 0.0f;
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        float breath = fastSinPhase(bs.breathPhase[i]) * 0.5f + 0.5f;
        glow[i] = bloomBreathFloor + breath * (bs.breathPeak[i] - bloomBreathFloor);
        glowSum += glow[i];
    }
    float glowMean = glowSum * (1.0f / (float)LEDS_PER_STRIP);
    // Audio energy → per-pixel overdrive added AFTER the knob/cap. Pixels above
    // the field mean blow through toward saturation; below-mean pixels go dark.
    float eStretch = (energyForce >= 0.0f) ? energyForce : audioEnergy5s;
    float audioDrive = bloomStretchGain * eStretch;   // 0 .. ~gain

    // Pass 2: ambient (knob-scaled) + audio overdrive (knob-independent).
    for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
        // Signed deviation from the field mean, normalized to ~[-1, +1.2].
        float w = clampf((glow[i] - glowMean) * 4.0f, -1.0f, 1.5f);
        // Ambient breathing: dim when the brightness knob is low.
        float ambient = fastGamma24(glow[i]) * bloomBrightnessCap * renderBrightness;
        // Audio punch is full-scale linear (NOT knob-scaled) so it blows through.
        float breathLin = clampf(ambient + audioDrive * w, 0.0f, bloomAudioCeil);

        bs.breathPhase[i] += dt / bs.breathPeriod[i];
        if (bs.breathPhase[i] >= 1.0f) bs.breathPhase[i] -= 1.0f;
        bs.hueT[i] += bs.hueDrift[i] * dt;
        if (bs.hueT[i] > 1.0f) bs.hueT[i] -= 1.0f;
        else if (bs.hueT[i] < 0.0f) bs.hueT[i] += 1.0f;

        float h = bs.hueT[i];
        float baseR = lerpf(bloomHueA_R, bloomHueB_R, h);
        float baseG = lerpf(bloomHueA_G, bloomHueB_G, h);
        float baseB = lerpf(bloomHueA_B, bloomHueB_B, h);

        float oR = baseR * breathLin + (baseR * oneMinusFF + BLOOM_FLASH_R * flashFrac) * flashLin;
        float oG = baseG * breathLin + (baseG * oneMinusFF + BLOOM_FLASH_G * flashFrac) * flashLin;
        float oB = baseB * breathLin + (baseB * oneMinusFF + BLOOM_FLASH_B * flashFrac) * flashLin;

        // Hysteresis noise gate with dormancy hold
        float maxCh = fmaxf(fmaxf(oR, oG), oB);
        float thresh = bs.gateOff[i] ? GATE_OFF_THRESH : GATE_ON_THRESH;
        if (maxCh < thresh) {
            oR = oG = oB = 0.0f;
            if (!bs.gateOff[i] && bs.dormancyRoll[i] < bloomDormancyFrac) {
                bs.gateOff[i] = true;
                bs.blackoutTimer[i] = 0.0f;
                bs.dormancyDur[i] = bloomDormancyMin
                    + randFloat() * (bloomDormancyMax - bloomDormancyMin);
            }
        } else if (bs.gateOff[i]) {
            bs.blackoutTimer[i] += dt;
            if (bs.blackoutTimer[i] < bs.dormancyDur[i]) {
                oR = oG = oB = 0.0f;
            } else {
                bs.gateOff[i] = false;
                bs.blackoutTimer[i] = 0.0f;
            }
        }

        // renderBrightness already folded into `ambient`; the audio overdrive is
        // intentionally NOT scaled by it so loud audio punches through a low knob.

        uint16_t t16R = (uint16_t)fminf(oR * 256.0f, 65535.0f);
        uint16_t t16G = (uint16_t)fminf(oG * 256.0f, 65535.0f);
        uint16_t t16B = (uint16_t)fminf(oB * 256.0f, 65535.0f);
        if ((t16R | t16G | t16B) == 0) {
            bs.ditherR[i] = bs.ditherG[i] = bs.ditherB[i] = 0;
        }
        uint8_t r8 = t16R >> 8;
        uint8_t g8 = t16G >> 8;
        uint8_t b8 = t16B >> 8;

        setPixel(s, i, r8, g8, b8);
    }
}

// ── Gravity particle (ported from bulb-fleet renderGravitySparkle) ─
// Particles fall under accelerometer tilt and splat as OKLCH glows.
// Stubbed at rest (ax=0) → no gravity → particles settle, gentle static.
// No per-effect cap (renderBrightness via setPixelScaled).
#define GS_PARTICLE_COUNT 7
#define GS_GRAVITY_SCALE  40.0f
#define GS_VELOCITY_DAMP  0.92f
#define GS_BOUNCE_REBOUND 0.5f
#define GS_SPLAT_RADIUS   2.5f
// Audio-driven breathing: speech (energy) springs each particle OUT from its
// origin toward the strip ends; silence relaxes it back home. Spread is the
// max outward displacement at full energy; stiff is the spring pull strength.
#define GS_SPREAD         (LEDS_PER_STRIP * 0.40f)
#define GS_SPRING_STIFF   55.0f

struct GsParticle {
    float pos;
    float vel;
    float bright;
    float hue;
    float origin;   // home position; particle springs out from / back to this
    float dir;      // outward direction (+1/-1), away from strip center
};
static float gsEnergyEma = 0.0f;   // smoothed speech energy driving the spread
static GsParticle gsParticles[NUM_STRIPS][GS_PARTICLE_COUNT];

static void resetGravity() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        // Seed positions from this strip's stick boundaries so the topology is
        // visible the instant gravity starts; particles then fall from there.
        float seedPos[GS_PARTICLE_COUNT];
        uint8_t nSeed = 0;
        for (uint8_t k = 0; k < STICK_COUNT && nSeed < GS_PARTICLE_COUNT; k++) {
            if (STICKS[k].strip != s) continue;
            seedPos[nSeed++] = (float)STICKS[k].start;
            if (nSeed < GS_PARTICLE_COUNT)
                seedPos[nSeed++] = (float)STICKS[k].end;
        }
        // Strips with no sticks (1, 3): fall back to an even spread.
        if (nSeed == 0) {
            for (uint8_t k = 0; k < GS_PARTICLE_COUNT; k++)
                seedPos[k] = (float)(LEDS_PER_STRIP - 1)
                    * (float)k / (float)(GS_PARTICLE_COUNT - 1);
            nSeed = GS_PARTICLE_COUNT;
        }

        const float center = (float)(LEDS_PER_STRIP - 1) * 0.5f;
        for (uint16_t i = 0; i < GS_PARTICLE_COUNT; i++) {
            float home = seedPos[i % nSeed];
            gsParticles[s][i].pos = home;
            gsParticles[s][i].origin = home;
            // Outward = away from the strip's midpoint, so speech expands the
            // whole field toward both ends. Particles exactly at center go up.
            gsParticles[s][i].dir = (home >= center) ? 1.0f : -1.0f;
            gsParticles[s][i].vel = 0.0f;
            gsParticles[s][i].bright = 1.0f;
            float baseHue = 256.0f * (float)i / (float)GS_PARTICLE_COUNT;
            gsParticles[s][i].hue = fmodf(baseHue + (float)STRIP_HUE_OFFSET[s], 256.0f);
        }
    }
}

static void renderGravity(float dt) {
    // Speech energy springs particles out from origin; silence relaxes them home.
    // gDt, not the speed-scaled dt: the ones knob shapes motion, never reaction
    // latency — sensing stays real-time like the rest of the audio path.
    gsEnergyEma += fminf(1.0f, gDt / 0.15f) * (fxEnergy - gsEnergyEma);
    float damp = fastDecay(GS_VELOCITY_DAMP, dt * 30.0f);

    const float maxPos = (float)(LEDS_PER_STRIP - 1);
    const float invTwoSigSq = 1.0f / (2.0f * GS_SPLAT_RADIUS * GS_SPLAT_RADIUS);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            fxAccR[i] = 0; fxAccG[i] = 0; fxAccB[i] = 0;
        }

        for (uint16_t i = 0; i < GS_PARTICLE_COUNT; i++) {
            GsParticle &p = gsParticles[s][i];

            // Target = origin pushed outward in proportion to speech energy.
            // Silence → target collapses to origin → spring pulls the particle home.
            float target = p.origin + p.dir * gsEnergyEma * GS_SPREAD;
            target = clampf(target, 0.0f, maxPos);
            p.vel += GS_SPRING_STIFF * (target - p.pos) * dt;   // spring toward target
            p.vel *= damp;                                       // damping for settle
            p.pos += p.vel * dt;

            // Home is a one-way wall: the underdamped spring would otherwise
            // slosh past the origin to the inward side on release. Arrest the
            // particle at home instead — outward stays bouncy, return settles.
            float outDisp = (p.pos - p.origin) * p.dir;
            if (outDisp < 0.0f) {
                p.pos = p.origin;
                if (p.vel * p.dir < 0.0f) p.vel = 0.0f;
            }

            if (p.pos < 0.0f) {
                p.pos = 0.0f;
                if (p.vel < 0.0f) p.vel = -p.vel * GS_BOUNCE_REBOUND;
            } else if (p.pos > maxPos) {
                p.pos = maxPos;
                if (p.vel > 0.0f) p.vel = -p.vel * GS_BOUNCE_REBOUND;
            }

            uint8_t hueIdx = (uint8_t)((uint32_t)p.hue & 0xFF);
            float colR = (float)oklchVarL[hueIdx][0];
            float colG = (float)oklchVarL[hueIdx][1];
            float colB = (float)oklchVarL[hueIdx][2];

            int center = (int)(p.pos + 0.5f);
            int lo = center - 3; if (lo < 0) lo = 0;
            int hi = center + 3; if (hi > (int)(LEDS_PER_STRIP - 1)) hi = LEDS_PER_STRIP - 1;
            for (int j = lo; j <= hi; j++) {
                float d = (float)j - p.pos;
                float w = expf(-(d * d) * invTwoSigSq);
                fxAccR[j] += colR * w;
                fxAccG[j] += colG * w;
                fxAccB[j] += colB * w;
            }
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float r = fxAccR[i];
            float g = fxAccG[i];
            float b = fxAccB[i];
            float maxCh = fmaxf(r, fmaxf(g, b));
            float bright = clampf(maxCh / 255.0f, 0.0f, 1.0f);
            float linBright = fastGamma24(bright);
            float norm = (bright > 0.001f) ? (linBright / bright) : 0.0f;
            setPixelScaled(s, i,
                clampf(r * norm, 0.0f, 255.0f),
                clampf(g * norm, 0.0f, 255.0f),
                clampf(b * norm, 0.0f, 255.0f));
        }
    }
}

// ── Sparkle syllable (ported from bulb-fleet) ───────────────────
// Onset-triggered LED ignition over a dim warm base. Stubbed: no onset,
// no tilt → renders the quiet warm-amber floor. No per-effect cap.
#define SPARKLE_DEADBAND 0.08f

static float syllSparkle[NUM_STRIPS][LEDS_PER_STRIP];
static float syllDecayArr[NUM_STRIPS][LEDS_PER_STRIP];
static float syllEnvelope = 0.0f;
static float syllCooldown = 0.0f;

static void resetSparkle() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            syllSparkle[s][i] = 0.0f;
            syllDecayArr[s][i] = 0.94f;
        }
    }
    syllEnvelope = 0.0f;
    syllCooldown = 0.0f;
}

static void renderSparkle(float dt) {
    float energy = fxEnergy;
    float onsetNorm = fxOnset;
    float angleDeg = fxAngleDeg;
    float tiltBlend = fxTiltBlend;

    // gDt, not the speed-scaled dt: the ones knob shapes motion, never
    // reaction latency — envelope and onset cooldown track speech in real
    // time at any speed (same rule as gravity's gsEnergyEma).
    float attackAlpha = fminf(1.0f, gDt / 0.030f);
    float decayAlpha  = fminf(1.0f, gDt / 0.400f);
    if (energy > syllEnvelope)
        syllEnvelope += attackAlpha * (energy - syllEnvelope);
    else
        syllEnvelope += decayAlpha * (energy - syllEnvelope);

    syllCooldown = fmaxf(0.0f, syllCooldown - gDt);

    if (onsetNorm > 0.1f && syllCooldown <= 0.0f) {
        syllCooldown = 0.060f;
        int nIgnite = (int)(LEDS_PER_STRIP * (0.25f + 0.25f * onsetNorm));
        float sparkVal = 0.6f + 0.4f * onsetNorm;
        for (uint8_t s = 0; s < NUM_STRIPS; s++) {
            static uint16_t indices[LEDS_PER_STRIP];
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) indices[i] = i;
            for (int i = 0; i < nIgnite; i++) {
                int j = i + (int)(xorshift32() % (LEDS_PER_STRIP - i));
                uint16_t tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
            for (int i = 0; i < nIgnite; i++) {
                syllSparkle[s][indices[i]] = sparkVal;
                syllDecayArr[s][indices[i]] = 0.92f + randFloat() * 0.05f;
            }
        }
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            syllSparkle[s][i] *= fastDecay(syllDecayArr[s][i], dt * 30.0f);

    float base = fminf(syllEnvelope, 0.15f);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float tiltR = 0, tiltG = 0, tiltB = 0;
        if (tiltBlend > 0.0f) {
            float hueFrac = (angleDeg - DEADZONE_DEG) / (MAX_ANGLE_DEG - DEADZONE_DEG);
            if (hueFrac < 0.0f) hueFrac = 0.0f;
            if (hueFrac > 1.0f) hueFrac = 1.0f;
            uint8_t hueIdx = (uint8_t)(((uint32_t)(hueFrac * 255)
                                         + STRIP_HUE_OFFSET[s]) & 0xFF);
            tiltR = (float)oklchVarL[hueIdx][0];
            tiltG = (float)oklchVarL[hueIdx][1];
            tiltB = (float)oklchVarL[hueIdx][2];
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float sp = syllSparkle[s][i];
            float bright = base + sp * (1.0f - base);
            if (bright < SPARKLE_DEADBAND) bright = 0.0f;

            float colR = 255.0f;
            float colG = 180.0f + (240.0f - 180.0f) * sp;
            float colB =  80.0f + (200.0f -  80.0f) * sp;

            if (tiltBlend > 0.0f) {
                colR = colR * (1.0f - tiltBlend) + tiltR * tiltBlend;
                colG = colG * (1.0f - tiltBlend) + tiltG * tiltBlend;
                colB = colB * (1.0f - tiltBlend) + tiltB * tiltBlend;
            }

            float wFold = 127.0f * (1.0f - tiltBlend);
            colR = fminf(255.0f, colR + wFold);
            colG = fminf(255.0f, colG + wFold);
            colB = fminf(255.0f, colB + wFold);

            float linBright = fastGamma24(bright);
            setPixelScaled(s, i,
                clampf(colR * linBright, 0.0f, 255.0f),
                clampf(colG * linBright, 0.0f, 255.0f),
                clampf(colB * linBright, 0.0f, 255.0f));
        }
    }
}

// ── Fire — audio-driven flame (ported from bulb-fleet renderFire) ───
// Base brightness rides speech energy (50 ms attack, 2 s decay; faint ember
// glow in silence). Onsets pump a flicker term on top. Sustained steady
// sound slowly "burns down" patches of LEDs toward ember-red (dropout) so
// held notes read as settling coals; transients re-ignite them. Color rides
// energy too: cool white when quiet → amber → red when loud; percussive-only
// input (onset with little energy) suppresses color toward white.
// Palette anchors are WS2811-corrected for these green-dominant LEDs (~2×
// red): G pulled well below the bulb-fleet values so amber doesn't read
// yellow and white doesn't read green. Eyeball-tune on hardware.
#define FIRE_FLICKER_SCALE  3.0f
#define FIRE_DEADBAND       0.08f
#define FIRE_DROPOUT_DEPTH  0.85f
// bulb-fleet shipped two slots (FIRE_MELD without dropout, FIRE_FLICKER
// with); we have one slot, so dropout is a compile-time toggle to compare.
static const bool FIRE_WITH_DROPOUT = true;

static float fireTime[NUM_STRIPS];
static float fireBaseBrightness = 0.0f;
static float fireFlickerIntensity = 0.0f;
static float fireColorEnergy = 0.0f;
static float firePrevEnergyForDeriv = 0.0f;
static float fireEnergyDerivSmooth = 0.0f;
static float fireDropoutAmount = 0.0f;

static void resetFire() {
    // Stagger phase per strip so flicker isn't lockstep.
    for (uint8_t s = 0; s < NUM_STRIPS; s++) fireTime[s] = (float)s * 1.37f;
    fireBaseBrightness = 0.0f;
    fireFlickerIntensity = 0.0f;
    fireColorEnergy = 0.0f;
    firePrevEnergyForDeriv = 0.0f;
    fireEnergyDerivSmooth = 0.0f;
    fireDropoutAmount = 0.0f;
}

static void renderFire(float dt) {
    // Audio-feature EMAs use gDt (real time), not the speed-scaled dt: the
    // ones knob shapes flame motion, never reaction latency — same convention
    // as gravity's gsEnergyEma.
    float energy = fxEnergy;
    float onset  = fxOnset;
    bool isSilent = energy < 0.001f;
    bool isPercussiveOnly = (!isSilent && energy < 0.15f && onset > 0.5f);

    float attackAlpha = fminf(1.0f, gDt / 0.050f);
    float decayAlpha  = fminf(1.0f, gDt / 2.0f);
    float targetBrightness = isSilent ? 0.25f : fmaxf(0.25f, energy);
    if (targetBrightness > fireBaseBrightness)
        fireBaseBrightness += attackAlpha * (targetBrightness - fireBaseBrightness);
    else
        fireBaseBrightness += decayAlpha * (targetBrightness - fireBaseBrightness);

    // fxOnset is discrete (zero between events, refractory-gated) where
    // bulb-fleet's onset was continuous; the 200 ms EMA turns each event
    // into a flicker pulse that decays back — pumps on attacks either way.
    float flickerAlpha = fminf(1.0f, gDt / 0.200f);
    float deltaTarget = isSilent ? 0.0f : onset;
    fireFlickerIntensity += flickerAlpha * (deltaTarget - fireFlickerIntensity);

    float dropoutAmount = 0.0f;
    if (FIRE_WITH_DROPOUT) {
        // Sustained sound = energy present but barely changing → coals settle.
        float energyDeriv = (energy - firePrevEnergyForDeriv) / fmaxf(gDt, 0.001f);
        firePrevEnergyForDeriv = energy;
        float derivAlpha = fminf(1.0f, gDt / 0.200f);
        fireEnergyDerivSmooth += derivAlpha * (energyDeriv - fireEnergyDerivSmooth);
        bool isSustaining = (!isSilent && energy > 0.05f
                             && fabsf(fireEnergyDerivSmooth) <= 0.5f);
        if (isSustaining)
            fireDropoutAmount = fminf(1.0f, fireDropoutAmount + gDt * 0.35f);
        else
            fireDropoutAmount = fmaxf(0.0f, fireDropoutAmount - gDt * 1.0f);
        dropoutAmount = fireDropoutAmount;
    }

    float colorAttack = fminf(1.0f, gDt / 0.080f);
    float colorDecay  = fminf(1.0f, gDt / 2.0f);
    float colorTarget;
    if (isPercussiveOnly)  colorTarget = 0.0f;
    else if (isSilent)     colorTarget = 0.3f;
    else                   colorTarget = fmaxf(0.3f, energy);
    if (colorTarget > fireColorEnergy)
        fireColorEnergy += colorAttack * (colorTarget - fireColorEnergy);
    else
        fireColorEnergy += colorDecay * (colorTarget - fireColorEnergy);

    // Base color from smoothed color-energy: white below the blend threshold,
    // amber through the mid range, red at RED_FULL and above.
    float ce = fireColorEnergy;
    const float WHITE_BLEND_THRESHOLD = 0.15f;
    const float RED_FULL = 0.5f;
    const float amberR = 255.0f, amberG = 75.0f, amberB = 5.0f;
    const float redR   = 200.0f, redG   = 15.0f, redB   = 0.0f;
    const float whiteR = 180.0f, whiteG = 85.0f, whiteB = 160.0f;
    float baseColR, baseColG, baseColB;
    if (ce < WHITE_BLEND_THRESHOLD) {
        float tw = 1.0f - (ce / WHITE_BLEND_THRESHOLD);
        baseColR = amberR * (1.0f - tw) + whiteR * tw;
        baseColG = amberG * (1.0f - tw) + whiteG * tw;
        baseColB = amberB * (1.0f - tw) + whiteB * tw;
    } else {
        float tr = (ce - WHITE_BLEND_THRESHOLD) / (RED_FULL - WHITE_BLEND_THRESHOLD);
        if (tr > 1.0f) tr = 1.0f;
        baseColR = amberR * (1.0f - tr) + redR * tr;
        baseColG = amberG * (1.0f - tr) + redG * tr;
        baseColB = amberB * (1.0f - tr) + redB * tr;
    }

    float base = fireBaseBrightness;
    if (base < FIRE_DEADBAND) base = 0.0f;
    float sScl = FIRE_FLICKER_SCALE;

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        fireTime[s] = fmodf(fireTime[s] + dt, 6283.1853f);
        float t = fireTime[s];
        // Per-strip spatial offset so noise patterns don't align.
        float sOff = (float)s * 17.0f;

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float fi = (float)i + sOff;

            float noise = fastSin(fi * 7.3f + t * 2.5f) *
                          fastSin(fi * 3.7f + t * 1.4f) * 0.5f + 0.5f;

            // Relative flicker grows as the base dims so quiet fire still
            // visibly dances; flicker term adds onset-driven sparkle on top.
            float noiseAmp = fmaxf(0.15f * sScl, 0.10f * sScl / fmaxf(base, 0.1f));
            float bright = base * (1.0f + noiseAmp * (noise - 0.5f))
                         + fireFlickerIntensity * (noise - 0.5f) * 0.25f * sScl;

            float perLedDim = 0.0f;
            float colorRedShift = 0.0f;
            if (FIRE_WITH_DROPOUT && dropoutAmount > 0.0f) {
                // Slow-moving resilience field: low-resilience LEDs burn down
                // first, dimming and red-shifting toward embers.
                float resilience = fastSin(fi * 13.7f + t * 0.3f) *
                                   fastSin(fi * 9.1f + t * 0.2f) * 0.5f + 0.5f;
                perLedDim = clampf(
                    (dropoutAmount - resilience * 0.7f) / 0.3f, 0.0f, 1.0f
                ) * FIRE_DROPOUT_DEPTH;
                colorRedShift = clampf(perLedDim / 0.3f, 0.0f, 1.0f);
            }

            bright *= (1.0f - perLedDim);
            bright = clampf(bright, 0.0f, 1.0f);

            float colR = baseColR * (1.0f - colorRedShift) + redR * colorRedShift;
            float colG = baseColG * (1.0f - colorRedShift) + redG * colorRedShift;
            float colB = baseColB * (1.0f - colorRedShift) + redB * colorRedShift;

            float linBright = fastGamma24(bright);
            setPixelScaled(s, i,
                clampf(colR * linBright, 0.0f, 255.0f),
                clampf(colG * linBright, 0.0f, 255.0f),
                clampf(colB * linBright, 0.0f, 255.0f));
        }
    }
}

// ── Nebula (ported from bulb-fleet) ─────────────────────────────
// Breathing blue→magenta background + warm-white drifting orbs.
// No audio dependency. dt is pre-scaled by effSpeed at the call site;
// the intrinsic 0.3 tuning factor is kept (it is not the speed knob).
#define NEBULA_MAX_ORBS       5
#define NEBULA_ORB_TAIL       30.0f
#define NEBULA_ORB_BASE_SPEED 0.45f
#define NEBULA_SPAWN_CHANCE   0.03f   // per-frame @30fps reference; dt-scaled at spawn
#define NEBULA_MIN_LIFETIME   200     // frames @60fps reference; /60 to seconds at spawn
#define NEBULA_MAX_LIFETIME   300
#define NEBULA_INTRINSIC_SPD  0.3f

struct NebOrb {
    float pos;
    float vel;
    float age;
    float lifetime;
    bool active;
};

static float nebulaTime = 0.0f;
static NebOrb nebOrbs[NUM_STRIPS][NEBULA_MAX_ORBS];
#define nebDecay fxField.neb   // shared per-effect field buffer

static void resetNebula() {
    nebulaTime = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++)
            nebOrbs[s][o].active = false;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            nebDecay[s][i] = 0.0f;
    }
}

static void renderNebula(float dt) {
    float spd = NEBULA_INTRINSIC_SPD;
    nebulaTime += dt * spd;
    float t = nebulaTime * 60.0f;

    float decayPerFrame = 1.0f - (1.0f / NEBULA_ORB_TAIL);
    float decayPerSec = powf(decayPerFrame, 60.0f);
    float decay = powf(decayPerSec, dt);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            nebDecay[s][i] *= decay;
            if (nebDecay[s][i] < 0.01f) nebDecay[s][i] = 0.0f;
        }

        float spawnRoll = (float)(xorshift32() & 0xFFFF) / 65536.0f;
        if (spawnRoll < NEBULA_SPAWN_CHANCE * dt * 30.0f) {
            for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++) {
                if (!nebOrbs[s][o].active) {
                    NebOrb &orb = nebOrbs[s][o];
                    orb.pos = randFloat() * (float)LEDS_PER_STRIP;
                    float dir = (xorshift32() & 1) ? 1.0f : -1.0f;
                    orb.vel = dir * NEBULA_ORB_BASE_SPEED * spd * (0.7f + randFloat() * 0.6f);
                    orb.age = 0.0f;
                    orb.lifetime = (NEBULA_MIN_LIFETIME
                        + (float)(xorshift32() % (NEBULA_MAX_LIFETIME - NEBULA_MIN_LIFETIME)))
                        / 60.0f;
                    orb.active = true;
                    break;
                }
            }
        }

        for (uint8_t o = 0; o < NEBULA_MAX_ORBS; o++) {
            NebOrb &orb = nebOrbs[s][o];
            if (!orb.active) continue;

            orb.age += dt;
            orb.pos += orb.vel * dt * 60.0f;
            orb.pos = fmodf(orb.pos + (float)LEDS_PER_STRIP, (float)LEDS_PER_STRIP);

            if (orb.age >= orb.lifetime) {
                orb.active = false;
                continue;
            }

            float lc = (float)orb.age / (float)orb.lifetime;
            float bright;
            if (lc < 0.4f) {
                float tf = lc / 0.4f;
                bright = tf * tf * (3.0f - 2.0f * tf);
            } else if (lc > 0.6f) {
                float tf = (1.0f - lc) / 0.4f;
                bright = tf * tf * (3.0f - 2.0f * tf);
            } else {
                bright = 1.0f;
            }

            int base = (int)orb.pos;
            int next = (base + 1) % LEDS_PER_STRIP;
            float frac = orb.pos - (float)base;
            float val0 = bright * 0.6f * (1.0f - frac);
            float val1 = bright * 0.6f * frac;
            if (base >= 0 && base < LEDS_PER_STRIP)
                nebDecay[s][base] = fminf(1.0f, nebDecay[s][base] + val0);
            if (next >= 0 && next < LEDS_PER_STRIP)
                nebDecay[s][next] = fminf(1.0f, nebDecay[s][next] + val1);
        }

        float sOff = (float)s * 0.167f;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float pos = (float)i / (float)LEDS_PER_STRIP;

            float breathing = 51.0f + 38.0f * fastSin((t * 0.0105f));
            float phase = pos + sOff + t * 0.006f;
            float spatial = 51.0f * (0.5f + 0.5f * fastSin(phase * 6.2832f));
            float bgBright = clampf(breathing + spatial, 0.0f, 153.0f) / 255.0f;

            float colorPhase = pos * 6.2832f + sOff * 6.2832f + t * 0.009f;
            float colorShift = 0.5f + 0.5f * fastSin(colorPhase);

            float bgR = (20.0f + colorShift * 235.0f) / 255.0f;
            float bgG = (30.0f - colorShift * 20.0f) / 255.0f;
            float bgB = (255.0f - colorShift * 125.0f) / 255.0f;

            float r = bgR * bgBright;
            float g = bgG * bgBright;
            float b = bgB * bgBright;

            float orbB = nebDecay[s][i];
            if (orbB > 0.01f) {
                r += orbB * 1.0f;
                g += orbB * 0.94f;
                b += orbB * 0.78f;
            }

            setPixelScaled(s, i,
                clampf(r, 0.0f, 1.0f) * 255.0f,
                clampf(g, 0.0f, 1.0f) * 255.0f,
                clampf(b, 0.0f, 1.0f) * 255.0f);
        }
    }
}

// ── Rainbow (OKLCH scroll, ported from bulb-fleet renderIdle) ───
// No audio dependency. Pure ambient hue scroll. dt is pre-scaled by
// effSpeed at the call site. No per-effect brightness cap — output
// scale is renderBrightness via setPixelScaled.
#define RAINBOW_SCROLL_SPEED  0.10f

static float rainbowPhase = 0.0f;

static void resetRainbow() {
    rainbowPhase = 0.0f;
}

static void renderRainbow(float dt) {
    rainbowPhase = fmodf(rainbowPhase + RAINBOW_SCROLL_SPEED * dt, 1.0f);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float stripOff = STRIP_HUE_OFFSET[s] / 256.0f;
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float pos = (float)i / (float)LEDS_PER_STRIP;
            float hue = fmodf(pos + rainbowPhase + stripOff, 1.0f);
            uint8_t idx = (uint8_t)(hue * 255.0f);
            setPixelScaled(s, i,
                (float)oklchVarL[idx][0],
                (float)oklchVarL[idx][1],
                (float)oklchVarL[idx][2]);
        }
    }
}

// ── Leaf wind (ported from bulb-fleet renderLeafWind) ───────────
// 1D drift along each strip: leaves blow from one end to the other, each LED
// lights by 1D distance to each leaf. Bulb-fleet's 2D topology.h is a hex map
// for a different installation, so tree-of-record collapses to a simple linear
// topology — LED index → position 0..1 (i / (LEDS_PER_STRIP-1)). Each leaf
// drifts independently per strip. No per-effect cap — output via setPixelScaled.
#define LW_MAX_LEAVES     8     // per strip
#define LW_GLOW_RADIUS    0.05f
#define LW_GLOW_SQ2       (2.0f * LW_GLOW_RADIUS * LW_GLOW_RADIUS)
#define LW_WIND_SPEED     0.35f
#define LW_SPAWN_INTERVAL 0.9f
#define LW_FADE_IN        0.4f
#define LW_VEL_TAU        0.22f   // velocity EMA time constant (sec); = orig 0.85/frame @30fps
#define LW_TURBULENCE     0.3f
// Launch gust. Was 0.25 / TC 2.5s — but flight takes ~2.5s, so the gust used to
// decay away mid-journey and the leaf braked ~11% toward the tip (measured).
// TC now 20s (>> flight) makes the gust a near-constant per-leaf speed offset:
// no tip-braking, leaves still ride different gusts. Magnitude trimmed 0.25→0.157
// so overall speed is unchanged (sim: 0.455→0.448).
#define LW_BOOST_SPEED    0.157f
#define LW_BOOST_TC       20.0f

static const uint8_t LW_PALETTE[][3] = {
    {255, 140, 20}, {240, 100, 10}, {220, 60, 5}, {200, 40, 10},
    {180, 30, 5},   {255, 180, 40}, {160, 25, 5},
};
#define LW_PALETTE_SIZE (sizeof(LW_PALETTE) / sizeof(LW_PALETTE[0]))

struct LwLeaf {
    float pos, vel, boost, age, brightness;
    uint8_t r, g, b;
    bool active;
};
static LwLeaf lwLeaves[NUM_STRIPS][LW_MAX_LEAVES];
static float lwTime = 0.0f;
static float lwSpawnTimer[NUM_STRIPS];

static float lwNoise1d(float pos, float t, int seed) {
    return (fastSin(pos * 0.4f + t * 0.3f + seed * 7.3f)
          * fastSin(pos * 0.17f - t * 0.19f + seed * 3.1f + (float)M_PI * 0.5f)
          + fastSin(pos * 0.09f + t * 0.13f + seed * 1.7f) * 0.5f) / 1.5f;
}

static void resetLeafWind() {
    lwTime = 0.0f;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        lwSpawnTimer[s] = randFloat() * LW_SPAWN_INTERVAL;
        for (int i = 0; i < LW_MAX_LEAVES; i++) lwLeaves[s][i].active = false;
    }
}

static void lwSpawnLeaf(uint8_t s) {
    for (int i = 0; i < LW_MAX_LEAVES; i++) {
        if (lwLeaves[s][i].active) continue;
        LwLeaf &lf = lwLeaves[s][i];
        lf.active = true;
        lf.pos = -0.05f;   // enter from LED 0 end, blow toward the tip
        lf.boost = LW_BOOST_SPEED * (0.5f + randFloat() * 0.5f);
        lf.vel = 0.0f;
        lf.age = 0.0f;
        lf.brightness = 0.0f;
        int ci = (int)(xorshift32() % LW_PALETTE_SIZE);
        lf.r = LW_PALETTE[ci][0];
        lf.g = LW_PALETTE[ci][1];
        lf.b = LW_PALETTE[ci][2];
        return;
    }
}

static void renderLeafWind(float dt) {
    lwTime += dt;
    float boostDecay = expf(-dt / LW_BOOST_TC);
    float lwAlpha = fminf(1.0f, dt / LW_VEL_TAU);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        lwSpawnTimer[s] += dt;
        while (lwSpawnTimer[s] >= LW_SPAWN_INTERVAL) {
            lwSpawnTimer[s] -= LW_SPAWN_INTERVAL;
            lwSpawnLeaf(s);
        }

        for (int i = 0; i < LW_MAX_LEAVES; i++) {
            if (!lwLeaves[s][i].active) continue;
            LwLeaf &lf = lwLeaves[s][i];

            float noise = lwNoise1d(lf.pos * 5.0f + (float)s * 3.0f, lwTime, i);
            float speedMult = fmaxf(0.1f, 1.0f + noise * LW_TURBULENCE);
            float force = LW_WIND_SPEED * speedMult;

            lf.vel = fmaxf(LW_WIND_SPEED * 0.4f, lf.vel + lwAlpha * (force - lf.vel));
            lf.boost *= boostDecay;
            lf.pos += (lf.vel + lf.boost) * dt;
            lf.age += dt;

            lf.brightness = (lf.age < LW_FADE_IN) ? (lf.age / LW_FADE_IN) : 1.0f;

            if (lf.pos > 1.05f) lf.active = false;
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float lpos = (float)i / (float)(LEDS_PER_STRIP - 1);
            float totalGlow = 0.0f, cr = 0.0f, cg = 0.0f, cb = 0.0f;

            for (int li = 0; li < LW_MAX_LEAVES; li++) {
                if (!lwLeaves[s][li].active) continue;
                LwLeaf &lf = lwLeaves[s][li];
                float d = lpos - lf.pos;
                float intensity = expf(-(d * d) / LW_GLOW_SQ2) * lf.brightness;
                if (intensity < 0.005f) continue;
                totalGlow += intensity;
                cr += intensity * lf.r;
                cg += intensity * lf.g;
                cb += intensity * lf.b;
            }

            if (totalGlow > 0.01f) {
                cr /= totalGlow; cg /= totalGlow; cb /= totalGlow;
                float bright = fminf(totalGlow, 1.0f);
                float linBright = fastGamma24(bright);
                setPixelScaled(s, i, cr * linBright, cg * linBright, cb * linBright);
            } else {
                setPixelScaled(s, i, 0.0f, 0.0f, 0.0f);
            }
        }
    }
}

// ── Creatures (ported from biolum_mixed) ────────────────────────
// Bloom + crawl creatures drift along a virtual buffer, one independent
// simulation per physical strip (no mirroring — every strip is its own world).
// Excitement is audio-driven: the speech-energy envelope (same signal as
// gravity's spread) stands in for the original accel shake — this install
// has no motion input, so the old shake/gyro-scroll feeds were retired.
// In silence the visuals fall back to pure ambient drift. Virtual buffer is
// 100 physical LEDs + a SCROLL_MARGIN each side (margin kept for the buffer
// layout even though gyro scroll is gone).
// No per-effect cap — output via setPixelScaled.
#define CR_NUM_UNITS     6     // one independent creature world per strip
#define CR_MAX_CREATURES 7
#define CR_MAX_ANIMS     6
#define CR_SCROLL_MARGIN 12
#define CR_VIRTUAL_LEDS  (LEDS_PER_STRIP + 2 * CR_SCROLL_MARGIN)
#define CR_SPAWN_MARGIN  5
#define CR_DESPAWN_MARGIN (-5)

#define CR_COLOR_R   0.0f
#define CR_COLOR_G 180.0f
#define CR_COLOR_B 220.0f

// --- Excitement params (buffer/drain machinery from biolum_mixed shake) ---
// Audio feed: smoothed speech energy above CR_AUDIO_THRESH stands in for the
// old accel excess-g. After the ceiling calibration, deliberate close speech
// rides ~0.7–1.0 and room music ~0.2, so 0.25 keeps music from exciting the
// creatures. CR_AUDIO_GAIN maps the max excess (~0.75) onto the ~2 scale the
// buffer gain/drain constants were tuned for.
#define CR_AUDIO_THRESH         0.25f
#define CR_AUDIO_GAIN           2.7f
#define CR_SHAKE_BUFFER_GAIN    0.8f
#define CR_SHAKE_BUFFER_MAX     0.5f
#define CR_SHAKE_DRAIN_RATE     0.8f
#define CR_SHAKE_BASE_FALL      0.5f
#define CR_SHAKE_FALL_SLOWDOWN  0.5f
#define CR_SHAKE_SCATTER_RADIUS 1.5f
#define CR_SHAKE_BRIGHT_BOOST   0.8f
#define CR_SHAKE_HUE_DRIFT      90.0f

#define CR_PULSE_EXPANSION_SPEED 3.3f
// Ratio doubled (was 1.9) → drift halved → each creature lives ~2x longer on
// screen. Pulse expansion speed is untouched, so animations look identical.
#define CR_PULSE_TO_DRIFT_RATIO  3.8f
#define CR_AVG_DRIFT  (CR_PULSE_EXPANSION_SPEED / CR_PULSE_TO_DRIFT_RATIO)
#define CR_DRIFT_SPREAD 0.33f
#define CR_DRIFT_MIN  (CR_AVG_DRIFT * (1.0f - CR_DRIFT_SPREAD))
#define CR_DRIFT_MAX  (CR_AVG_DRIFT * (1.0f + CR_DRIFT_SPREAD))

#define CR_BLOOM_RADIUS        3.0f
#define CR_BLOOM_EDGE_SOFTNESS 0.8f
#define CR_BLOOM_RISE          2.0f
#define CR_BLOOM_HOLD          1.5f
#define CR_BLOOM_FALL          5.0f
#define CR_BLOOM_TOTAL  (CR_BLOOM_RISE + CR_BLOOM_HOLD + CR_BLOOM_FALL)
#define CR_BLOOM_EMIT_LO 1.2f
#define CR_BLOOM_EMIT_HI 2.8f

#define CR_CRAWL_RADIUS         8.0f
#define CR_CRAWL_PULSE_LIFETIME (CR_CRAWL_RADIUS / CR_PULSE_EXPANSION_SPEED)
#define CR_CRAWL_PULSE_FADE     1.4f
#define CR_CRAWL_TAIL_DECAY     4.0f
#define CR_CRAWL_EMIT_LO        1.2f
#define CR_CRAWL_EMIT_HI        2.8f

enum CrKind : uint8_t { CR_KIND_BLOOM, CR_KIND_CRAWL };

struct CrAnim { float age; bool active; };
struct CrCreature {
    float    pos, vel;
    CrKind   kind;
    bool     alive;
    float    emitTimer;
    CrAnim   anims[CR_MAX_ANIMS];
    float    hueOffset, hueSweep;
};
struct CrUnit {
    CrCreature creatures[CR_MAX_CREATURES];
    float bufR[CR_VIRTUAL_LEDS], bufG[CR_VIRTUAL_LEDS], bufB[CR_VIRTUAL_LEDS];
};
static CrUnit crUnits[CR_NUM_UNITS];

// --- Global excitement state (driven by processInteraction) ---
static float crShakeLevel   = 0.0f;   // smooth 0→1 controlling visual effects
static float crShakeBuffer  = 0.0f;   // energy buffer (spiky input → smooth drain)
static float crShakeTime    = 0.0f;   // accumulated seconds excited
static bool  crShakeActive  = false;
static float crShakeCooldown = 2.0f;
static bool  crLifecycleFrozen = false;
static float crShakeHueDrift = 0.0f;
static float crEnergyEma    = 0.0f;   // smoothed speech energy (audio feed)

static inline float crRandf(float lo, float hi) {
    return lo + randFloat() * (hi - lo);
}

static void crProcessInteraction(float dt) {
    // gDt for the envelope, not the speed-scaled dt: the ones knob shapes the
    // creatures' motion, never reaction latency (gravity convention).
    crEnergyEma += fminf(1.0f, gDt / 0.15f) * (fxEnergy - crEnergyEma);
    float excess = (crEnergyEma - CR_AUDIO_THRESH) * CR_AUDIO_GAIN;
    if (excess > 0.0f) {
        crShakeActive = true;
        crShakeBuffer += excess * CR_SHAKE_BUFFER_GAIN * dt;
        if (crShakeBuffer > CR_SHAKE_BUFFER_MAX) crShakeBuffer = CR_SHAKE_BUFFER_MAX;
        crShakeTime += dt;
    } else {
        crShakeActive = false;
    }

    float drained = fminf(crShakeBuffer, CR_SHAKE_DRAIN_RATE * sqrtf(crShakeBuffer) * dt);
    crShakeBuffer -= drained;

    if (crShakeActive) {
        crShakeLevel = fminf(1.0f, crShakeLevel + drained);
    } else {
        float fallRate = CR_SHAKE_BASE_FALL / (1.0f + crShakeTime * CR_SHAKE_FALL_SLOWDOWN);
        crShakeLevel -= fallRate * dt;
        if (crShakeLevel <= 0.0f) { crShakeLevel = 0.0f; crShakeTime = 0.0f; }
    }

    if (crShakeLevel > 0.2f) crShakeCooldown = 0.0f;
    else                     crShakeCooldown += dt;
    crLifecycleFrozen = (crShakeLevel > 0.2f || crShakeCooldown < 0.3f);

    crShakeHueDrift += crShakeLevel * CR_SHAKE_HUE_DRIFT * dt;
    if (crShakeHueDrift > 360.0f) crShakeHueDrift -= 360.0f;
}

static void crInitCreature(CrCreature &c) {
    c.kind = (xorshift32() & 1) ? CR_KIND_BLOOM : CR_KIND_CRAWL;
    float speed = crRandf(CR_DRIFT_MIN, CR_DRIFT_MAX);
    c.pos = crRandf(CR_SPAWN_MARGIN, CR_VIRTUAL_LEDS - CR_SPAWN_MARGIN);
    c.vel = (xorshift32() & 1) ? speed : -speed;
    c.alive = true;
    c.emitTimer = (c.kind == CR_KIND_BLOOM)
        ? crRandf(CR_BLOOM_EMIT_LO, CR_BLOOM_EMIT_HI)
        : crRandf(CR_CRAWL_EMIT_LO, CR_CRAWL_EMIT_HI);
    for (int i = 0; i < CR_MAX_ANIMS; i++) c.anims[i].active = false;
    c.hueOffset = crRandf(0.0f, 360.0f);
    c.hueSweep = crRandf(60.0f, 90.0f);
}

static void resetCreatures() {
    for (int p = 0; p < CR_NUM_UNITS; p++)
        for (int c = 0; c < CR_MAX_CREATURES; c++)
            crInitCreature(crUnits[p].creatures[c]);
}

static void crEmitAnim(CrCreature &c) {
    for (int i = 0; i < CR_MAX_ANIMS; i++) {
        if (!c.anims[i].active) {
            c.anims[i].active = true;
            c.anims[i].age = 0.0f;
            return;
        }
    }
}

static void crUpdateCreature(CrCreature &c, float dt) {
    if (!c.alive) return;
    c.pos += c.vel * dt;
    if (c.pos < CR_DESPAWN_MARGIN || c.pos > CR_VIRTUAL_LEDS - CR_DESPAWN_MARGIN) {
        c.alive = false;
        return;
    }
    c.emitTimer -= dt;
    if (c.emitTimer <= 0.0f) {
        crEmitAnim(c);
        if (c.kind == CR_KIND_BLOOM) {
            float base = CR_BLOOM_TOTAL + crRandf(CR_BLOOM_EMIT_LO, CR_BLOOM_EMIT_HI);
            c.emitTimer = base * crRandf(0.33f, 2.0f);
        } else {
            c.emitTimer = CR_CRAWL_PULSE_LIFETIME + CR_CRAWL_PULSE_FADE
                          + crRandf(CR_CRAWL_EMIT_LO, CR_CRAWL_EMIT_HI);
        }
    }
    float maxAge = (c.kind == CR_KIND_BLOOM) ? CR_BLOOM_TOTAL
                   : (CR_CRAWL_PULSE_LIFETIME + CR_CRAWL_PULSE_FADE);
    for (int i = 0; i < CR_MAX_ANIMS; i++) {
        if (!c.anims[i].active) continue;
        c.anims[i].age += dt;
        if (c.anims[i].age >= maxAge) c.anims[i].active = false;
    }
}

// Shake-aware color: blend base teal toward a per-position rainbow.
static inline void crShakeColor(const CrCreature &c, float relOut,
                                float &pixR, float &pixG, float &pixB) {
    pixR = CR_COLOR_R; pixG = CR_COLOR_G; pixB = CR_COLOR_B;
    if (crShakeLevel > 0.01f) {
        float colorT = crShakeLevel;
        float hueFrac = (relOut + 1.0f) * 0.5f;
        float hue = fmodf(c.hueOffset + crShakeHueDrift + hueFrac * c.hueSweep, 360.0f);
        float rR, rG, rB;
        hsvToRgb(hue, 1.0f, 1.0f, rR, rG, rB);
        pixR = CR_COLOR_R * (1.0f - colorT) + rR * colorT;
        pixG = CR_COLOR_G * (1.0f - colorT) + rG * colorT;
        pixB = CR_COLOR_B * (1.0f - colorT) + rB * colorT;
    }
}

static void crRenderBloom(const CrCreature &c, float *bufR, float *bufG, float *bufB) {
    float center = c.pos;
    float halfWidth = CR_BLOOM_RADIUS + CR_BLOOM_EDGE_SOFTNESS;
    float totalSpan = halfWidth * (1.0f + crShakeLevel * CR_SHAKE_SCATTER_RADIUS);
    int lo = (int)(center - totalSpan - 1); if (lo < 0) lo = 0;
    int hi = (int)(center + totalSpan + 1); if (hi > CR_VIRTUAL_LEDS - 1) hi = CR_VIRTUAL_LEDS - 1;

    float brightMult = 1.0f + crShakeLevel * CR_SHAKE_BRIGHT_BOOST;

    for (int a = 0; a < CR_MAX_ANIMS; a++) {
        if (!c.anims[a].active) continue;
        float age = c.anims[a].age;
        float envelope;
        if (age < CR_BLOOM_RISE) {
            envelope = age / CR_BLOOM_RISE;
        } else if (age < CR_BLOOM_RISE + CR_BLOOM_HOLD) {
            envelope = 1.0f;
        } else {
            float fallT = (age - CR_BLOOM_RISE - CR_BLOOM_HOLD) / CR_BLOOM_FALL;
            envelope = fmaxf(0.0f, 1.0f - fallT);
            envelope *= envelope;
        }
        if (envelope < 0.003f) continue;

        for (int i = lo; i <= hi; i++) {
            float relOut = ((float)i - center) / totalSpan;
            if (fabsf(relOut) > 1.0f) continue;
            float srcDist = fabsf(relOut) * halfWidth;
            float spatial = (srcDist <= CR_BLOOM_RADIUS) ? 1.0f
                : expf(-(srcDist - CR_BLOOM_RADIUS) / CR_BLOOM_EDGE_SOFTNESS);

            float gapFade = 1.0f;
            if (crShakeLevel > 0.01f) {
                float gapSize = 0.3f * crShakeLevel;
                if (fabsf(relOut) < gapSize) {
                    gapFade = fabsf(relOut) / gapSize;
                    gapFade *= gapFade;
                }
            }

            float br = envelope * spatial * brightMult * gapFade;
            if (br < 0.003f) continue;

            float pixR, pixG, pixB;
            crShakeColor(c, relOut, pixR, pixG, pixB);
            float normR = pixR / 255.0f * br;
            float normG = pixG / 255.0f * br;
            float normB = pixB / 255.0f * br;
            bufR[i] = bufR[i] + normR - bufR[i] * normR;
            bufG[i] = bufG[i] + normG - bufG[i] * normG;
            bufB[i] = bufB[i] + normB - bufB[i] * normB;
        }
    }
}

static void crRenderCrawl(const CrCreature &c, float *bufR, float *bufG, float *bufB) {
    float center = c.pos;
    float halfWidth = CR_CRAWL_RADIUS + CR_CRAWL_TAIL_DECAY;
    float totalSpan = halfWidth * (1.0f + crShakeLevel * CR_SHAKE_SCATTER_RADIUS);
    int lo = (int)(center - totalSpan - 1); if (lo < 0) lo = 0;
    int hi = (int)(center + totalSpan + 1); if (hi > CR_VIRTUAL_LEDS - 1) hi = CR_VIRTUAL_LEDS - 1;

    float brightMult = 1.0f + crShakeLevel * CR_SHAKE_BRIGHT_BOOST;

    for (int a = 0; a < CR_MAX_ANIMS; a++) {
        if (!c.anims[a].active) continue;
        float t = fminf(c.anims[a].age / CR_CRAWL_PULSE_LIFETIME, 1.0f);
        float radius = t * CR_CRAWL_RADIUS;
        float fade = 1.0f;
        if (c.anims[a].age > CR_CRAWL_PULSE_LIFETIME) {
            float fadeT = (c.anims[a].age - CR_CRAWL_PULSE_LIFETIME) / CR_CRAWL_PULSE_FADE;
            fade = fmaxf(0.0f, 1.0f - fadeT);
            fade *= fade;
        }
        for (int i = lo; i <= hi; i++) {
            float relOut = ((float)i - center) / totalSpan;
            if (fabsf(relOut) > 1.0f) continue;
            float srcDist = fabsf(relOut) * halfWidth;
            if (srcDist > radius) continue;
            float behind = radius - srcDist;
            float br = expf(-behind / CR_CRAWL_TAIL_DECAY);
            if (behind < 1.0f) br *= behind;

            float gapFade = 1.0f;
            if (crShakeLevel > 0.01f) {
                float gapSize = 0.3f * crShakeLevel;
                if (fabsf(relOut) < gapSize) {
                    gapFade = fabsf(relOut) / gapSize;
                    gapFade *= gapFade;
                }
            }

            br *= fade * brightMult * gapFade;
            if (br < 0.003f) continue;

            float pixR, pixG, pixB;
            crShakeColor(c, relOut, pixR, pixG, pixB);
            float normR = pixR / 255.0f * br;
            float normG = pixG / 255.0f * br;
            float normB = pixB / 255.0f * br;
            bufR[i] = bufR[i] + normR - bufR[i] * normR;
            bufG[i] = bufG[i] + normG - bufG[i] * normG;
            bufB[i] = bufB[i] + normB - bufB[i] * normB;
        }
    }
}

static void renderCreatures(float dt) {
    crProcessInteraction(dt);

    for (int p = 0; p < CR_NUM_UNITS; p++) {
        CrUnit &ps = crUnits[p];
        memset(ps.bufR, 0, sizeof(ps.bufR));
        memset(ps.bufG, 0, sizeof(ps.bufG));
        memset(ps.bufB, 0, sizeof(ps.bufB));

        for (int c = 0; c < CR_MAX_CREATURES; c++) {
            if (!crLifecycleFrozen) crUpdateCreature(ps.creatures[c], dt);
            if (!ps.creatures[c].alive) continue;
            if (ps.creatures[c].kind == CR_KIND_BLOOM)
                crRenderBloom(ps.creatures[c], ps.bufR, ps.bufG, ps.bufB);
            else
                crRenderCrawl(ps.creatures[c], ps.bufR, ps.bufG, ps.bufB);
        }
        for (int c = 0; c < CR_MAX_CREATURES; c++)
            if (!ps.creatures[c].alive) crInitCreature(ps.creatures[c]);

        // One unit drives one physical strip — no mirroring.
        uint8_t s = p;
        for (int i = 0; i < LEDS_PER_STRIP; i++) {
            int src = CR_SCROLL_MARGIN + i;
            float vR = clampf(ps.bufR[src], 0.0f, 1.0f);
            float vG = clampf(ps.bufG[src], 0.0f, 1.0f);
            float vB = clampf(ps.bufB[src], 0.0f, 1.0f);
            // Gamma the scalar brightness and rescale channels proportionally
            // (gravity's pattern). Per-channel gamma — what the original port
            // did — shifts hue on mixed colors as they dim (doctrine rule:
            // gamma the brightness multiplier, never individual channels).
            float bright = fmaxf(vR, fmaxf(vG, vB));
            float norm = (bright > 0.001f) ? fastGamma24(bright) / bright : 0.0f;
            setPixelScaled(s, i, vR * norm * 255.0f,
                                 vG * norm * 255.0f,
                                 vB * norm * 255.0f);
        }
    }
}

// ── Light Through (stormy sunset, slot 10) ──────────────────────
// Storm-ceiling base field (slate↔indigo) with warm bloom-patches that swell
// open then seal, like sun breaking through cloud. 6 strips are radial spokes
// from center: r = index/(LEDS-1) (0=trunk, 1=edge), spoke s at angle s×60°.
// Spawn/churn/growth all scale with fxDt (speed knob folds in upstream).
// Routes through setPixelScaled like every other effect.
#define LT_MAX_PATCHES   6
#define LT_SPAWN_PERIOD  2.2f   // mean seconds between patches (Poisson)
#define LT_CHURN_RATE    0.12f

// Palette (0–255 linear).
#define LT_SLATE_R   50.0f
#define LT_SLATE_G   55.0f
#define LT_SLATE_B   80.0f
#define LT_INDIGO_R  35.0f
#define LT_INDIGO_G  35.0f
#define LT_INDIGO_B  60.0f
#define LT_AMBER_R  240.0f
#define LT_AMBER_G  180.0f
#define LT_AMBER_B   90.0f
#define LT_PEACH_R  220.0f
#define LT_PEACH_G  160.0f
#define LT_PEACH_B  110.0f
#define LT_SALMON_R 200.0f
#define LT_SALMON_G 140.0f
#define LT_SALMON_B 100.0f

struct LtPatch {
    float rc;        // radial center 0.15–0.85
    float cosTc;     // precomputed cos/sin of angular center
    float sinTc;
    float maxR;      // 0.25–0.7
    float life;      // 1.5–3.5 s
    float age;
    bool  active;
};
static LtPatch ltPatches[LT_MAX_PATCHES];
static float ltTime = 0.0f;
static float ltSpawnTimer = 0.0f;
static float ltSpokeCos[NUM_STRIPS];
static float ltSpokeSin[NUM_STRIPS];

// Layered-sine pseudo-noise in 0..1 (cheap, no Perlin).
static inline float ltNoise(float x) {
    return 0.5f + 0.5f * fastSin(x * 2.3f + fastSin(x * 1.7f + ltTime * 0.08f));
}

static void resetLightThrough() {
    ltTime = 0.0f;
    ltSpawnTimer = 0.0f;
    for (int i = 0; i < LT_MAX_PATCHES; i++) ltPatches[i].active = false;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float ang = (float)s * (60.0f * (float)M_PI / 180.0f);
        ltSpokeCos[s] = cosf(ang);
        ltSpokeSin[s] = sinf(ang);
    }
}

static void ltSpawnPatch() {
    for (int i = 0; i < LT_MAX_PATCHES; i++) {
        if (ltPatches[i].active) continue;
        LtPatch &p = ltPatches[i];
        p.rc = 0.15f + randFloat() * 0.70f;        // [0.15, 0.85]
        float tc = randFloat() * 2.0f * (float)M_PI;
        p.cosTc = cosf(tc);
        p.sinTc = sinf(tc);
        p.maxR = 0.25f + randFloat() * 0.45f;       // [0.25, 0.7]
        p.life = 1.5f + randFloat() * 2.0f;         // [1.5, 3.5]
        p.age = 0.0f;
        p.active = true;
        return;
    }
}

static void renderLightThrough(float dt) {
    ltTime += dt;

    // Poisson spawn: probability dt/period per frame.
    ltSpawnTimer += dt;
    float spawnProb = dt / LT_SPAWN_PERIOD;
    if (randFloat() < spawnProb) ltSpawnPatch();

    for (int i = 0; i < LT_MAX_PATCHES; i++) {
        if (!ltPatches[i].active) continue;
        ltPatches[i].age += dt;
        if (ltPatches[i].age >= ltPatches[i].life) ltPatches[i].active = false;
    }

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float sc = ltSpokeCos[s];
        float ss = ltSpokeSin[s];
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float r = (float)i / (float)(LEDS_PER_STRIP - 1);

            // Layer A — storm ceiling.
            float density = ltNoise(r * 3.0f + 0.7f * (float)s + ltTime * LT_CHURN_RATE);
            float centerLift = lerpf(1.0f, 0.85f, r);
            float outR = lerpf(LT_SLATE_R, LT_INDIGO_R, density) * centerLift;
            float outG = lerpf(LT_SLATE_G, LT_INDIGO_G, density) * centerLift;
            float outB = lerpf(LT_SLATE_B, LT_INDIGO_B, density) * centerLift;

            // LED planar position on the spoke.
            float px = r * sc;
            float py = r * ss;

            // Layer B — warm bloom-patches (sum where they overlap).
            for (int pi = 0; pi < LT_MAX_PATCHES; pi++) {
                if (!ltPatches[pi].active) continue;
                LtPatch &p = ltPatches[pi];
                float tau = p.age / p.life;
                float radius = p.maxR * fastSin((float)M_PI * tau);
                if (radius < 0.001f) continue;

                float dx = px - p.rc * p.cosTc;
                float dy = py - p.rc * p.sinTc;
                float d = sqrtf(dx * dx + dy * dy);

                float g = d / (0.6f * radius);
                float a = expf(-(g * g));
                if (a < 0.004f) continue;

                // Color: amber→peach→salmon by d/radius.
                float dr = clampf(d / radius, 0.0f, 1.0f);
                float pcR, pcG, pcB;
                if (dr < 0.5f) {
                    float t = dr / 0.5f;
                    pcR = lerpf(LT_AMBER_R, LT_PEACH_R, t);
                    pcG = lerpf(LT_AMBER_G, LT_PEACH_G, t);
                    pcB = lerpf(LT_AMBER_B, LT_PEACH_B, t);
                } else {
                    float t = (dr - 0.5f) / 0.5f;
                    pcR = lerpf(LT_PEACH_R, LT_SALMON_R, t);
                    pcG = lerpf(LT_PEACH_G, LT_SALMON_G, t);
                    pcB = lerpf(LT_PEACH_B, LT_SALMON_B, t);
                }

                outR = outR * (1.0f - a) + pcR * a + 0.15f * a * 255.0f;
                outG = outG * (1.0f - a) + pcG * a + 0.15f * a * 255.0f;
                outB = outB * (1.0f - a) + pcB * a;
            }

            setPixelScaled(s, i,
                clampf(outR, 0.0f, 255.0f),
                clampf(outG, 0.0f, 255.0f),
                clampf(outB, 0.0f, 255.0f));
        }
    }
}

// ── Worley collision (ported from audio-reactive worley_voronoi.py) ─
// Voronoi seeds on springs, evenly spaced at rest. Speech onsets kick
// them outward from strip center; the cell boundaries (F2−F1 proximity)
// flare bright and drift back as the springs settle. Dark cell
// interiors, OKLCH rainbow per cell.
#define WORLEY_SEEDS      5
#define WORLEY_SPRING_K   12.0f   // spring stiffness (1/s²)
#define WORLEY_DAMPING    7.0f    // ~critically damped vs SPRING_K
#define WORLEY_BOUNDARY_W 5.0f    // boundary falloff width (px)
#define WORLEY_AMBIENT    0.12f   // boundary glow with no audio
#define WORLEY_CELL_DIM   0.03f   // faint cell-interior floor
#define WORLEY_ENV_DECAY  0.88f   // flash decay per frame @30fps reference

static float worleyKick = 80.0f;  // px/s impulse per unit onset strength
static float worleyPos[NUM_STRIPS][WORLEY_SEEDS];
static float worleyVel[NUM_STRIPS][WORLEY_SEEDS];
static float worleyRest[NUM_STRIPS][WORLEY_SEEDS];
static float worleyEnv = 0.0f;

static void resetWorley() {
    float spacing = (float)LEDS_PER_STRIP / (float)(WORLEY_SEEDS + 1);
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t k = 0; k < WORLEY_SEEDS; k++) {
            // Per-strip jitter so the 6 strips don't read as copies.
            worleyRest[s][k] = spacing * (float)(k + 1)
                             + (randFloat() - 0.5f) * spacing * 0.3f;
            worleyPos[s][k] = worleyRest[s][k];
            worleyVel[s][k] = 0.0f;
        }
    }
    worleyEnv = 0.0f;
}

static void renderWorley(float dt) {
    // gDt for the flash envelope: the ones knob shapes the spring motion,
    // never reaction latency.
    if (fxOnset > 0.1f) {
        const float center = (float)(LEDS_PER_STRIP - 1) * 0.5f;
        for (uint8_t s = 0; s < NUM_STRIPS; s++)
            for (uint8_t k = 0; k < WORLEY_SEEDS; k++) {
                float dir = (worleyPos[s][k] > center) ? 1.0f : -1.0f;
                worleyVel[s][k] += dir * fxOnset * worleyKick;
            }
        worleyEnv = fmaxf(worleyEnv, fxOnset);
    }
    worleyEnv *= fastDecay(WORLEY_ENV_DECAY, gDt * 30.0f);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t k = 0; k < WORLEY_SEEDS; k++) {
            float disp = worleyPos[s][k] - worleyRest[s][k];
            float acc = -WORLEY_SPRING_K * disp - WORLEY_DAMPING * worleyVel[s][k];
            worleyVel[s][k] += acc * dt;
            worleyPos[s][k] += worleyVel[s][k] * dt;
            if (worleyPos[s][k] < 0.5f) {
                worleyPos[s][k] = 0.5f;
                worleyVel[s][k] *= -0.3f;
            } else if (worleyPos[s][k] > (float)LEDS_PER_STRIP - 0.5f) {
                worleyPos[s][k] = (float)LEDS_PER_STRIP - 0.5f;
                worleyVel[s][k] *= -0.3f;
            }
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float px = (float)i + 0.5f;
            float f1 = 1e9f, f2 = 1e9f;
            uint8_t nearest = 0;
            for (uint8_t k = 0; k < WORLEY_SEEDS; k++) {
                float d = fabsf(px - worleyPos[s][k]);
                if (d < f1) { f2 = f1; f1 = d; nearest = k; }
                else if (d < f2) { f2 = d; }
            }
            float boundary = expf(-(f2 - f1) / WORLEY_BOUNDARY_W);
            float bright = (WORLEY_AMBIENT + (1.0f - WORLEY_AMBIENT) * worleyEnv)
                         * boundary;
            bright = fmaxf(bright, WORLEY_CELL_DIM);

            uint8_t hueIdx = (uint8_t)((nearest * 51 + STRIP_HUE_OFFSET[s]) & 0xFF);
            float linBright = fastGamma24(bright);
            setPixelScaled(s, i,
                (float)oklchVarL[hueIdx][0] * linBright,
                (float)oklchVarL[hueIdx][1] * linBright,
                (float)oklchVarL[hueIdx][2] * linBright);
        }
    }
}

// ── Heat diffusion (ported from audio-reactive heat_diffusion.py) ──
// 1D heat equation per strip: speech energy injects heat at a slowly
// patrolling point, diffusion spreads it, cooling sinks it. Blackbody
// ramp black→deep red→orange→white; sustained loud speech reaches
// white-hot at the source (steady-state peak ≈ rate / (2·√(α·c))).
// A passive sparkle overlay drifts on top: 1–3 px wide, fading in and
// out over a few seconds at nominal speed, soft warm white.
#define HEAT_ALPHA       40.0f   // diffusivity (px²/s)
#define HEAT_COOLING     0.9f    // heat sink rate (1/s)
#define HEAT_ROT_PERIOD  18.0f   // s per strip traversal at nominal speed
#define HEAT_MAX_STEP    (0.4f / HEAT_ALPHA)  // explicit-Euler stability bound
#define HEAT_SPARKS      4       // passive sparkle slots per strip

static float heatInjRate = 14.0f;  // heat/s at fxEnergy = 1
#define heatT fxField.heat         // shared per-effect field buffer
static float heatInjPos[NUM_STRIPS];
static float heatInjDir[NUM_STRIPS];

struct HeatSpark {
    uint16_t pos;     // left edge
    uint8_t  width;   // 1–3 px
    float    phase;   // 0–1 through the fade-in/out cycle
    float    period;  // seconds for the full in+out
    float    idle;    // seconds until (re)spawn
    bool     active;
};
static HeatSpark heatSparks[NUM_STRIPS][HEAT_SPARKS];

static void resetHeatDiffusion() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) heatT[s][i] = 0.0f;
        heatInjPos[s] = randFloat() * (float)LEDS_PER_STRIP;
        heatInjDir[s] = (xorshift32() & 1) ? 1.0f : -1.0f;
        for (uint8_t k = 0; k < HEAT_SPARKS; k++) {
            heatSparks[s][k].active = false;
            heatSparks[s][k].idle = randFloat() * 6.0f;
        }
    }
}

// Blackbody-ish ramp; channels saturate in sequence so T>1 reads white.
static inline void heatRamp(float T, float &r, float &g, float &b) {
    r = clampf(T * 3.0f, 0.0f, 1.0f);
    g = clampf((T - 0.30f) * 1.6f, 0.0f, 1.0f);
    b = clampf((T - 0.80f) * 3.0f, 0.0f, 1.0f);
}

static void renderHeatDiffusion(float dt) {
    static float lap[LEDS_PER_STRIP];

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        float *T = heatT[s];

        // Patrol the injection point (motion → speed-scaled dt).
        heatInjPos[s] += heatInjDir[s]
                       * ((float)LEDS_PER_STRIP / HEAT_ROT_PERIOD) * dt;
        if (heatInjPos[s] < 0.0f) {
            heatInjPos[s] = 0.0f;
            heatInjDir[s] = 1.0f;
        } else if (heatInjPos[s] > (float)(LEDS_PER_STRIP - 1)) {
            heatInjPos[s] = (float)(LEDS_PER_STRIP - 1);
            heatInjDir[s] = -1.0f;
        }

        // Inject heat from speech energy. gDt, not the speed-scaled dt:
        // how strongly audio is expressed must not change with the ones
        // knob (same rule as gravity/sparkle).
        float inject = heatInjRate * fxEnergy * gDt;
        if (inject > 0.0f) {
            int c = (int)heatInjPos[s];
            T[c] += inject * 0.5f;
            if (c > 0)                  T[c - 1] += inject * 0.25f;
            if (c < LEDS_PER_STRIP - 1) T[c + 1] += inject * 0.25f;
        }

        // Diffuse + cool, substepped to stay inside the stability bound
        // when the speed knob stretches dt.
        float rem = dt;
        while (rem > 0.0f) {
            float step = fminf(rem, HEAT_MAX_STEP);
            rem -= step;
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
                float left  = (i > 0) ? T[i - 1] : T[i];
                float right = (i < LEDS_PER_STRIP - 1) ? T[i + 1] : T[i];
                lap[i] = left + right - 2.0f * T[i];
            }
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
                T[i] += (HEAT_ALPHA * lap[i] - HEAT_COOLING * T[i]) * step;
                if (T[i] < 0.0f) T[i] = 0.0f;
            }
        }

        // Passive sparkle lifecycle (ambient motion → speed-scaled dt).
        for (uint8_t k = 0; k < HEAT_SPARKS; k++) {
            HeatSpark &sp = heatSparks[s][k];
            if (!sp.active) {
                sp.idle -= dt;
                if (sp.idle <= 0.0f) {
                    sp.active = true;
                    sp.pos = (uint16_t)(xorshift32() % (LEDS_PER_STRIP - 2));
                    sp.width = 1 + (uint8_t)(xorshift32() % 3);
                    sp.period = 2.5f + randFloat() * 2.5f;
                    sp.phase = 0.0f;
                }
            } else {
                sp.phase += dt / sp.period;
                if (sp.phase >= 1.0f) {
                    sp.active = false;
                    sp.idle = 2.0f + randFloat() * 6.0f;
                }
            }
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float r, g, b;
            heatRamp(T[i], r, g, b);

            for (uint8_t k = 0; k < HEAT_SPARKS; k++) {
                HeatSpark &sp = heatSparks[s][k];
                if (!sp.active) continue;
                if (i < sp.pos || i >= sp.pos + sp.width) continue;
                float env = fastSin((float)M_PI * sp.phase);
                env = env * env * 0.30f;   // modest: sin² in/out, peak 0.30
                r += env;
                g += env * 0.92f;          // soft warm white
                b += env * 0.72f;
            }

            float bright = fmaxf(r, fmaxf(g, b));
            if (bright > 1.0f) {
                float inv = 1.0f / bright;
                r *= inv; g *= inv; b *= inv;
                bright = 1.0f;
            }
            float norm = (bright > 0.001f) ? fastGamma24(bright) / bright : 0.0f;
            setPixelScaled(s, i, r * norm * 255.0f, g * norm * 255.0f,
                           b * norm * 255.0f);
        }
    }
}

// ── Polycule rainbow (ported from audio-reactive polycule_rainbow.py) ─
// Five soft particles per strip with rainbow-gradient tails: hue cycles
// across each glow and drifts slowly over time. Particles bounce off
// strip ends and shove each other apart on collision. Purely ambient —
// no audio input by design (the Python original's RMS pulse is dropped).
#define POLY_PARTICLES  5
#define POLY_TRAVERSE_S 5.0f   // s end-to-end at nominal speed
#define POLY_SPEED_VAR  0.5f   // ±50% per-particle speed variation
#define POLY_COLLIDE_R  6.0f   // px
#define POLY_GLOW_SIGMA 2.5f   // px gaussian glow
#define POLY_WINDOW     8      // render window ±px (glow < 0.01 beyond)

static float polyPos[NUM_STRIPS][POLY_PARTICLES];
static float polyVel[NUM_STRIPS][POLY_PARTICLES];
static float polyTime = 0.0f;

static void resetPolycule() {
    polyTime = 0.0f;
    float baseSpeed = (float)(LEDS_PER_STRIP - 1) / POLY_TRAVERSE_S;
    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t k = 0; k < POLY_PARTICLES; k++) {
            polyPos[s][k] = (float)LEDS_PER_STRIP
                          * ((float)k + 0.5f) / (float)POLY_PARTICLES;
            float spd = baseSpeed * (1.0f - POLY_SPEED_VAR
                                     + randFloat() * 2.0f * POLY_SPEED_VAR);
            polyVel[s][k] = (k & 1) ? -spd : spd;
        }
    }
}

static void renderPolycule(float dt) {
    polyTime += dt;
    const float maxPos = (float)(LEDS_PER_STRIP - 1);

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        for (uint8_t k = 0; k < POLY_PARTICLES; k++) {
            polyPos[s][k] += polyVel[s][k] * dt;
            if (polyPos[s][k] > maxPos) {
                polyPos[s][k] = 2.0f * maxPos - polyPos[s][k];
                polyVel[s][k] = -polyVel[s][k];
            } else if (polyPos[s][k] < 0.0f) {
                polyPos[s][k] = -polyPos[s][k];
                polyVel[s][k] = -polyVel[s][k];
            }
        }

        // Collision: sort the 5 by position, shove overlapping neighbors
        // apart (matches the Python original — reverse, not momentum swap).
        uint8_t order[POLY_PARTICLES];
        for (uint8_t k = 0; k < POLY_PARTICLES; k++) order[k] = k;
        for (uint8_t a = 1; a < POLY_PARTICLES; a++) {
            uint8_t o = order[a];
            int b = (int)a - 1;
            while (b >= 0 && polyPos[s][order[b]] > polyPos[s][o]) {
                order[b + 1] = order[b];
                b--;
            }
            order[b + 1] = o;
        }
        for (uint8_t a = 0; a + 1 < POLY_PARTICLES; a++) {
            uint8_t lo = order[a], hi = order[a + 1];
            float gap = polyPos[s][hi] - polyPos[s][lo];
            if (gap < POLY_COLLIDE_R) {
                polyVel[s][lo] = -fabsf(polyVel[s][lo]);
                polyVel[s][hi] =  fabsf(polyVel[s][hi]);
                float push = (POLY_COLLIDE_R - gap) * 0.5f + 0.5f;
                polyPos[s][lo] = fmaxf(0.0f, polyPos[s][lo] - push);
                polyPos[s][hi] = fminf(maxPos, polyPos[s][hi] + push);
            }
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            fxAccR[i] = 0.0f; fxAccG[i] = 0.0f; fxAccB[i] = 0.0f;
        }

        for (uint8_t k = 0; k < POLY_PARTICLES; k++) {
            int center = (int)polyPos[s][k];
            int lo = center - POLY_WINDOW; if (lo < 0) lo = 0;
            int hi = center + POLY_WINDOW; if (hi > (int)maxPos) hi = (int)maxPos;
            for (int i = lo; i <= hi; i++) {
                float dist = fabsf((float)i - polyPos[s][k]);
                float glow = expf(-(dist * dist)
                                  / (2.0f * POLY_GLOW_SIGMA * POLY_GLOW_SIGMA));
                if (glow < 0.01f) continue;
                uint8_t hueIdx = (uint8_t)(((uint32_t)(k * 51)
                                  + (uint32_t)(dist * 15.0f)
                                  + (uint32_t)(polyTime * 25.0f)
                                  + STRIP_HUE_OFFSET[s]) & 0xFF);
                fxAccR[i] += (float)oklchVarL[hueIdx][0] * glow;
                fxAccG[i] += (float)oklchVarL[hueIdx][1] * glow;
                fxAccB[i] += (float)oklchVarL[hueIdx][2] * glow;
            }
        }

        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
            float r = fxAccR[i], g = fxAccG[i], b = fxAccB[i];
            float m = fmaxf(r, fmaxf(g, b));
            if (m > 255.0f) {
                float inv = 255.0f / m;
                r *= inv; g *= inv; b *= inv;
                m = 255.0f;
            }
            float bright = m * (1.0f / 255.0f);
            float norm = (bright > 0.001f) ? fastGamma24(bright) / bright : 0.0f;
            setPixelScaled(s, i, r * norm, g * norm, b * norm);
        }
    }
}

// ── Black out all strips (reserved/off slots) ───────────────────
static void renderOff() {
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
            setPixel(s, i, 0, 0, 0);
}

// ── Effect reset dispatch (called on effect change) ─────────────
static void resetEffect(uint8_t fx) {
    resetFxDither();
    switch (fx) {
        case FX_AMBIENT_BLOOM:
            for (uint8_t s = 0; s < NUM_STRIPS; s++) resetBloomStrip(bloom[s]);
            break;
        case FX_FIRE:
            resetFire();
            break;
        case FX_HEAT_DIFFUSION:
            resetHeatDiffusion();
            break;
        case FX_WORLEY:
            resetWorley();
            break;
        case FX_GRAVITY_PARTICLE:
            resetGravity();
            break;
        case FX_SPARKLE_SYLLABLE:
            resetSparkle();
            break;
        case FX_CREATURES:
            resetCreatures();
            break;
        case FX_POLYCULE:
            resetPolycule();
            break;
        case FX_LEAF_WIND:
            resetLeafWind();
            break;
        case FX_LIGHT_THROUGH:
            resetLightThrough();
            break;
        case FX_NEBULA:
            resetNebula();
            break;
        case FX_RAINBOW:
            resetRainbow();
            break;
        default:
            break;
    }
}

// ── Runtime parameter table ──────────────────────────────────────
struct Param {
    const char *name;
    enum Type { F32 } type;
    void *ptr;
    float lo, hi;
};

static const Param PARAMS[] = {
    { "BRIGHTNESS_CAP",   Param::F32, &bloomBrightnessCap,  0.05f, 1.0f  },
    { "GLOBAL_BRIGHTNESS",Param::F32, &globalBrightness,    0.0f,  1.0f  },
    { "SPEED_SCALE",      Param::F32, &speedScale,          0.2f,  3.0f  },
    { "BUFFER_DRAIN",     Param::F32, &bloomBufferDrain,    0.5f,  20.0f },
    { "FLASH_DECAY_RATE", Param::F32, &bloomFlashDecayRate, 0.2f,  10.0f },
    { "BREATH_FLOOR",     Param::F32, &bloomBreathFloor,    0.0f,  0.5f  },
    { "DORMANCY_MIN",     Param::F32, &bloomDormancyMin,   0.0f,  60.0f },
    { "DORMANCY_MAX",     Param::F32, &bloomDormancyMax,   0.0f,  60.0f },
    { "DORMANCY_FRAC",    Param::F32, &bloomDormancyFrac,  0.0f,  1.0f  },
    { "ONSET_REFRAC",     Param::F32, &onsetRefrac,        0.02f, 1.0f  },
    { "ONSET_RISEFRAC",   Param::F32, &onsetRiseFrac,      0.05f, 4.0f  },
    { "STRETCH_GAIN",     Param::F32, &bloomStretchGain,   0.0f,  6.0f  },
    { "AUDIO_CEIL",       Param::F32, &bloomAudioCeil,     1.0f,  4.0f  },
    { "ENERGY5S_TAU",     Param::F32, &energy5sTau,        0.5f,  20.0f },
    { "ENERGY_FORCE",     Param::F32, &energyForce,       -1.0f,  1.0f  },
    { "HEAT_INJ_RATE",    Param::F32, &heatInjRate,        1.0f,  60.0f },
    { "WORLEY_KICK",      Param::F32, &worleyKick,        10.0f, 300.0f },
    { "AUDIO_VOL",        Param::F32, &audioOutVol,        0.0f,  1.0f  },
};
static const size_t PARAM_COUNT = sizeof(PARAMS) / sizeof(PARAMS[0]);

static void dumpParam(const Param &p) {
    Serial.printf("[PARAM] %s=%.4f\n", p.name, *(float*)p.ptr);
}

static void dumpAllParams() {
    for (size_t i = 0; i < PARAM_COUNT; i++) dumpParam(PARAMS[i]);
}

static void setParamFromLine(const char *kv) {
    while (*kv == ' ') kv++;
    const char *eq = strchr(kv, '=');
    if (!eq) { Serial.println("[PARAM] bad syntax"); return; }
    size_t nameLen = (size_t)(eq - kv);
    const char *valStr = eq + 1;
    for (size_t i = 0; i < PARAM_COUNT; i++) {
        const Param &p = PARAMS[i];
        if (strlen(p.name) == nameLen && strncmp(p.name, kv, nameLen) == 0) {
            float v = atof(valStr);
            if (v < p.lo) v = p.lo;
            if (v > p.hi) v = p.hi;
            *(float*)p.ptr = v;
            dumpParam(p);
            return;
        }
    }
    Serial.printf("[PARAM] unknown key\n");
}

static void processLine(const char *line, uint8_t len) {
    if (len >= 2 && line[0] == '!' && line[1] == 'P') {
        if (len >= 3 && line[2] == '?') { dumpAllParams(); return; }
        if (len >= 4) { setParamFromLine(line + 2); return; }
    }
}

static char serialBuf[80];
static uint8_t serialBufLen = 0;

static void parseSerialCommands() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialBufLen > 0) {
                serialBuf[serialBufLen] = '\0';
                processLine(serialBuf, serialBufLen);
                serialBufLen = 0;
            }
        } else if (c >= 32 && c < 127 && serialBufLen < sizeof(serialBuf) - 1) {
            serialBuf[serialBufLen++] = c;
        }
    }
}

// ── Setup ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);

    prngState = esp_random();
    if (prngState == 0) prngState = 1;

    strip0.Begin(); strip1.Begin(); strip2.Begin();
    strip3.Begin(); strip4.Begin(); strip5.Begin();
    strip0.ClearTo(RgbColor(0)); strip1.ClearTo(RgbColor(0)); strip2.ClearTo(RgbColor(0));
    strip3.ClearTo(RgbColor(0)); strip4.ClearTo(RgbColor(0)); strip5.ClearTo(RgbColor(0));
    showAll();

    for (uint8_t s = 0; s < NUM_STRIPS; s++) {
        resetBloomStrip(bloom[s]);
    }

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init failed");
        return;
    }
    esp_now_register_recv_cb(onReceive);

    // Some core/IDF builds reset the WiFi channel during esp_now_init — re-assert
    // it so RX stays on the sender's channel.
    esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);

    // Register the broadcast peer so broadcast frames from the BS-26 sender are
    // accepted (mirrors the sender's broadcast peer setup).
    static const uint8_t BROADCAST_ADDR[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, BROADCAST_ADDR, 6);
    peer.channel = FIXED_CHANNEL;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    recInit();

    Serial.printf("Tree-of-record ready — ch=%u\n", FIXED_CHANNEL);
    Serial.printf("  Bloom: 6 strips × %u LEDs (ambient, BS-26 driven)\n", LEDS_PER_STRIP);
}

// ── Main loop ────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();
    static uint32_t lastRenderMs = 0;
    float dt = (lastRenderMs > 0) ? (now - lastRenderMs) / 1000.0f : (1.0f / SENSOR_HZ);
    if (dt > 0.1f) dt = 0.1f;
    gDt = dt;
    lastRenderMs = now;

    // Decode sensor packets → effect feature globals (real-time dt).
    processSensors(dt);

    // ── Map BS-26 knobs to effect parameters ────────────────────
    // BS-26 live → knobs drive the effect. No signal → serial PARAMS.
    bool bs26Live = (bs26LastMs > 0 && (int32_t)(now - bs26LastMs) < (int32_t)HEARTBEAT_TIMEOUT_MS);
    bool bs26EverSeen = (bs26LastMs > 0);
    // Last knob-derived brightness/speed, held across brief packet gaps so a
    // dropout doesn't snap brightness to the serial-param default. The other
    // knob params (hue/sat/decouter/effect/acOn) are persistent globals and
    // simply retain their last value when a frame is skipped.
    static float bs26LastBrightness = 1.0f;
    static float bs26LastSpeed      = 1.0f;

    float effBrightness = globalBrightness;
    float effSpeed      = speedScale;
    if (bs26Live) {
        Bs26State s;
        portENTER_CRITICAL(&bs26Mux);
        memcpy(&s, (const void*)&bs26, sizeof(s));
        portEXIT_CRITICAL(&bs26Mux);

        // tens (10-position knob) → brightness; 0 = fully off
        uint8_t tens = s.tens > 9 ? 9 : s.tens;
        if (tens == 0) {
            effBrightness = 0.0f;
        } else {
            effBrightness = powf((float)tens / 9.0f, 2.5f);
        }

        // ones (0–11) → speed. Fine log resolution 0.2→1.0 across positions 0–8
        // (ratio ≈1.22/step), then compressed octave doublings 2/4/8 at the top.
        static const float ONES_SPEED[12] = {
            0.20f, 0.24f, 0.30f, 0.37f, 0.45f, 0.55f, 0.67f, 0.82f, 1.00f,
            2.00f, 4.00f, 8.00f
        };
        uint8_t spd = s.ones > 11 ? 11 : s.ones;
        effSpeed = ONES_SPEED[spd];

        // hundreds (0–4) → recording slot select
        recSlot = s.hundreds > 4 ? 4 : s.hundreds;

        // decmid (0–9) → hue rotation, decinner (0–9) → saturation
        uint8_t hue = s.decmid > 9 ? 9 : s.decmid;
        uint8_t sat = s.decinner > 9 ? 9 : s.decinner;
        if (hue != paletteHueIdx || sat != paletteSatIdx) {
            paletteHueIdx = hue;
            paletteSatIdx = sat;
            applyPalette(paletteHueIdx, paletteSatIdx);
        }

        // decouter (0–11) → effect select
        currentEffect = s.decouter >= FX_COUNT ? FX_COUNT - 1 : s.decouter;

        // AC switch = record, DC switch = playback, both LEVEL (on = active).
        // Record has priority over playback. Releasing a switch re-arms it.
        bool acOn = s.ac, dcOn = s.dc;
        if (!acOn) recAcLatch = false;     // released → re-armed
        if (!dcOn) recDcLatch = false;

        if (recState == REC_RECORDING) {
            if (!acOn) recStopRecording();                 // switch off → stop
        } else if (recState == REC_PLAYING) {
            if (acOn && !recAcLatch) {                     // record overrides playback
                recStopPlayback();
                recStartRecording();
                recAcLatch = true;
            } else if (!dcOn) {
                recStopPlayback();                         // switch off → stop
            }
        } else {  // REC_IDLE
            if (acOn && !recAcLatch) {
                recStartRecording();
                recAcLatch = true;
            } else if (dcOn && !recDcLatch) {
                recStartPlayback();
                recDcLatch = true;
            }
        }

        bs26LastBrightness = effBrightness;
        bs26LastSpeed      = effSpeed;
    } else if (bs26EverSeen) {
        // Brief packet gap — hold the last knob-derived values rather than
        // reverting to the serial-param defaults (which would spike brightness).
        effBrightness = bs26LastBrightness;
        effSpeed      = bs26LastSpeed;
    }

    bloomDormancyFrac = 0.0f;

    // ── Recording: sample frames onto a fixed REC_FPS wall-clock grid ──
    // A RecFrame is a *param snapshot* (effect/brightness/palette + companded
    // audio bytes), not pixels — playback re-renders the effect live. Sampling
    // at a fixed rate instead of one-per-loop decouples the recording from the
    // render rate, so frame index == elapsed time. That's what lets playback
    // lock the frame to the audio clock no matter how heavy the effect is.
    if (recState == REC_RECORDING && recBuffer) {
        RecFrame rf;
        rf.tens = (uint8_t)(effBrightness * 255);
        rf.ones = (uint8_t)(effSpeed * 64);
        rf.hundreds = recSlot;
        rf.decouter = currentEffect;
        rf.decmid = paletteHueIdx;
        rf.decinner = paletteSatIdx;

        // Capture the filtered audio drive (companded RMS bytes) so silent
        // replay stays faithful. Mirror the live audio path: zero the bytes
        // when the packet is stale or the sender flagged the mic muted (bit0).
        rf.rms_mean = 0;
        rf.rms_max  = 0;
        if (audioLastMs > 0 && (int32_t)(now - audioLastMs) < (int32_t)SENSOR_TIMEOUT_MS) {
            AudioPacketV1 au;
            memcpy(&au, (const void*)&audioPkt, sizeof(au));
            if (!(au.flags & 0x01)) {   // bit0 = mic muted
                rf.rms_mean = au.rms_mean;
                rf.rms_max  = au.rms_max;
            }
        }

        // Emit zero, one, or several copies to keep the grid filled: loop faster
        // than REC_FPS → skip until the next tick; slower → backfill duplicates.
        recFrameAcc += dt;
        const float frameDt = 1.0f / REC_FPS;
        while (recFrameAcc >= frameDt && recFrameCount < REC_MAX_FRAMES) {
            recFrameAcc -= frameDt;
            recBuffer[recFrameCount++] = rf;
        }
        if (recFrameCount >= REC_MAX_FRAMES) recStopRecording();
    }

    // Drain received waveform samples to the slot's /rec_N.audio (8 kHz/8-bit).
    // Runs every loop while recording, independent of the 60 fps frame capture.
    if (recState == REC_RECORDING) recAudioDrain();

    // ── Playback: drive effect params from recorded frames ──
    if (recState == REC_PLAYING && recBuffer && recFrameCount > 0) {
        // Advance + output audio first: it's the master clock. For waveform slots
        // playbackAudioTick also re-derives fxEnergy/fxOnset and returns true; it
        // loops the file itself at EOF. Silent slots return false → we fall back
        // to the envelope bytes and a wall-clock frame counter.
        bool haveAudio = playbackAudioTick(dt);

        if (haveAudio) {
            // Lock the frame index to audio position: frame == time × REC_FPS.
            // Both were captured for the same wall-clock span (fixed-rate frames,
            // 8 kHz audio), so this can't drift regardless of render load.
            uint32_t pos = playAudioFile.position();
            recPlayIdx = (uint32_t)((float)pos / REC_AUDIO_RATE * REC_FPS);
            if (recPlayIdx >= recFrameCount) recPlayIdx = recFrameCount - 1;
        } else {
            recPlayClock += dt * REC_FPS;                 // frames at fixed REC_FPS
            recPlayIdx = (uint32_t)recPlayClock;
            if (recPlayIdx >= recFrameCount) { recPlayClock = 0.0f; recPlayIdx = 0; }
        }

        RecFrame &rf = recBuffer[recPlayIdx];
        effBrightness = (float)rf.tens / 255.0f;
        effSpeed = (float)rf.ones / 64.0f;
        currentEffect = rf.decouter >= FX_COUNT ? FX_COUNT - 1 : rf.decouter;
        uint8_t hue = rf.decmid;
        uint8_t sat = rf.decinner;
        if (hue != paletteHueIdx || sat != paletteSatIdx) {
            paletteHueIdx = hue;
            paletteSatIdx = sat;
            applyPalette(paletteHueIdx, paletteSatIdx);
        }
        if (!haveAudio)
            audioDeriveFeatures(rf.rms_mean, rf.rms_max, dt);
    }

    renderBrightness = effBrightness;
    renderFloor = effBrightness * 0.2f;

    // tens=0 → all off (blank and skip rendering)
    if (effBrightness == 0.0f) {
        for (uint8_t s = 0; s < NUM_STRIPS; s++)
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++)
                setPixel(s, i, 0, 0, 0);
        showAll();
        return;
    }

    // Reset effect state on change, but debounce so spinning the ones knob
    // through intermediate slots (e.g. gravity, whose reset does a white seed
    // flash) doesn't fire their resets in passing. Only the settled selection
    // resets.
    static uint8_t pendingEffect = 0xFF;
    static uint32_t effectSettleMs = 0;
    if (currentEffect != pendingEffect) {
        pendingEffect = currentEffect;
        effectSettleMs = now;
    }
    if (currentEffect != prevEffect &&
        (prevEffect == 0xFF || now - effectSettleMs >= 250)) {
        resetEffect(currentEffect);
        prevEffect = currentEffect;
    }

    uint32_t t0 = micros();

    // ── Render the selected effect on all 6 strips ──────────────
    // Per-effect speed normalization so the ones knob feels consistent.
    static const float SPEED_SCALE[] = {
        1.4f,   // FX_AMBIENT_BLOOM
        1.0f,   // FX_FIRE
        1.0f,   // FX_HEAT_DIFFUSION
        1.0f,   // FX_WORLEY
        0.5f,   // FX_GRAVITY_PARTICLE
        1.0f,   // FX_SPARKLE_SYLLABLE
        1.5f,   // FX_CREATURES
        1.0f,   // FX_POLYCULE
        0.25f,  // FX_LEAF_WIND
        1.0f,   // FX_LIGHT_THROUGH
        1.5f,   // FX_NEBULA
        2.0f,   // FX_RAINBOW
    };
    float fxDt = dt * effSpeed * SPEED_SCALE[currentEffect];
    switch (currentEffect) {
        case FX_AMBIENT_BLOOM:
            for (uint8_t s = 0; s < NUM_STRIPS; s++) renderBloomStrip(s, fxDt);
            break;
        case FX_FIRE:
            renderFire(fxDt);
            break;
        case FX_HEAT_DIFFUSION:
            renderHeatDiffusion(fxDt);
            break;
        case FX_WORLEY:
            renderWorley(fxDt);
            break;
        case FX_GRAVITY_PARTICLE:
            renderGravity(fxDt);
            break;
        case FX_SPARKLE_SYLLABLE:
            renderSparkle(fxDt);
            break;
        case FX_CREATURES:
            renderCreatures(fxDt);
            break;
        case FX_POLYCULE:
            renderPolycule(fxDt);
            break;
        case FX_LEAF_WIND:
            renderLeafWind(fxDt);
            break;
        case FX_LIGHT_THROUGH:
            renderLightThrough(fxDt);
            break;
        case FX_NEBULA:
            renderNebula(fxDt);
            break;
        case FX_RAINBOW:
            renderRainbow(fxDt);
            break;
        default:
            renderOff();
            break;
    }

    uint32_t t1 = micros();
    showAll();
    uint32_t t2 = micros();

    static uint32_t renderUsAccum = 0, showUsAccum = 0;
    renderUsAccum += (t1 - t0);
    showUsAccum   += (t2 - t1);

    parseSerialCommands();

    // Status logging
    static uint32_t lastLogMs = 0;
    static uint32_t frameCount = 0;
    frameCount++;
    if (now - lastLogMs > 2000) {
        float fps = frameCount * 1000.0f / (now - lastLogMs);
        float avgRenderUs = (float)renderUsAccum / frameCount;
        float avgShowUs   = (float)showUsAccum / frameCount;
        Serial.printf("  FPS=%.1f  render=%.0fus  show=%.0fus  total=%.0fus\n",
                      fps, avgRenderUs, avgShowUs, avgRenderUs + avgShowUs);
        RgbColor px = strip0.GetPixelColor(50);
        Serial.printf("  [px50] R=%u G=%u B=%u\n", px.R, px.G, px.B);
        if (currentEffect == FX_AMBIENT_BLOOM) {
            // Field span on strip 0: min/max per-pixel max-channel. Opens/closes
            // with the audio-energy contrast stretch — objective spread readout.
            uint8_t bMin = 255, bMax = 0;
            for (uint16_t i = 0; i < LEDS_PER_STRIP; i++) {
                RgbColor c = strip0.GetPixelColor(i);
                uint8_t m = c.R; if (c.G > m) m = c.G; if (c.B > m) m = c.B;
                if (m < bMin) bMin = m;
                if (m > bMax) bMax = m;
            }
            float eDbg = (energyForce >= 0.0f) ? energyForce : audioEnergy5s;
            Serial.printf("  [bloom] span=%u..%u e5s=%.3f force=%.2f stretch=%.2f\n",
                          bMin, bMax, audioEnergy5s, energyForce,
                          1.0f + bloomStretchGain * eDbg);
        }
        Serial.printf("  [bs26] %s pkts=%lu fx=%u bright=%.3f speed=%.2f hue=%u sat=%u dorm=%.2f"
                      " | ac=%d dc=%d slot=%u rec=%s\n",
                      bs26Live ? "LIVE" : "----",
                      (unsigned long)bs26PktCount,
                      currentEffect, renderBrightness, effSpeed,
                      paletteHueIdx, paletteSatIdx, bloomDormancyFrac,
                      bs26.ac ? 1 : 0, bs26.dc ? 1 : 0, bs26.hundreds,
                      recState == REC_RECORDING ? "REC" :
                      recState == REC_PLAYING   ? "PLAY" : "idle");
        bool aLive = accelLastMs && (int32_t)(now - accelLastMs) < (int32_t)SENSOR_TIMEOUT_MS;
        bool gLive = gyroLastMs  && (int32_t)(now - gyroLastMs)  < (int32_t)SENSOR_TIMEOUT_MS;
        bool auLive = audioLastMs && (int32_t)(now - audioLastMs) < (int32_t)SENSOR_TIMEOUT_MS;
        Serial.printf("  [sns] accel=%s(%lu) gyro=%s(%lu) audio=%s(%lu) "
                      "energy=%.3f onset=%.3f tilt=%.2f@%.0f crExcite=%.2f\n",
                      aLive ? "L" : "-", (unsigned long)accelPktCount,
                      gLive ? "L" : "-", (unsigned long)gyroPktCount,
                      auLive ? "L" : "-", (unsigned long)audioPktCount,
                      fxEnergy, fxOnset, fxTiltBlend, fxAngleDeg,
                      crShakeLevel);
        frameCount = 0;
        renderUsAccum = 0;
        showUsAccum = 0;
        lastLogMs = now;
    }
}
