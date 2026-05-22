// Road-bulbs effect simulator.
// Reads binary SensorPackets from stdin, outputs RGBW frames to stdout.
// Also writes per-frame diagnostics to stderr as JSON lines.
//
// Usage:
//   ./sim --effect gravity < recording.bin > frames.bin
//   ./sim --effect sparkle --json < recording.bin > /dev/null 2> diag.jsonl

#include "effects.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char* algName(Algorithm a) {
    switch (a) {
        case ALG_SPARKLE_BURST:   return "sparkle";
        case ALG_FIRE_MELD:       return "fire";
        case ALG_FIRE_FLICKER:    return "flicker";
        case ALG_QUIET_BLOOM:     return "bloom";
        case ALG_GRAVITY_PARTICLE: return "gravity";
        case ALG_SPARKLE_SYLLABLE: return "syllable";
    }
    return "unknown";
}

static Algorithm parseAlg(const char* name) {
    if (!strcmp(name, "gravity"))  return ALG_GRAVITY_PARTICLE;
    if (!strcmp(name, "sparkle"))  return ALG_SPARKLE_BURST;
    if (!strcmp(name, "fire"))     return ALG_FIRE_MELD;
    if (!strcmp(name, "flicker"))  return ALG_FIRE_FLICKER;
    if (!strcmp(name, "bloom"))    return ALG_QUIET_BLOOM;
    if (!strcmp(name, "syllable")) return ALG_SPARKLE_SYLLABLE;
    fprintf(stderr, "Unknown effect: %s\n", name);
    exit(1);
    return ALG_GRAVITY_PARTICLE;
}

int main(int argc, char** argv) {
    Algorithm alg = ALG_GRAVITY_PARTICLE;
    float fps = 25.0f;
    bool jsonDiag = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--effect") && i+1 < argc) {
            alg = parseAlg(argv[++i]);
        } else if (!strcmp(argv[i], "--fps") && i+1 < argc) {
            fps = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--json")) {
            jsonDiag = true;
        } else if (!strcmp(argv[i], "--help")) {
            fprintf(stderr, "Usage: sim [--effect name] [--fps N] [--json] < packets.bin > frames.bin\n");
            fprintf(stderr, "Effects: gravity, sparkle, fire, flicker, bloom, syllable\n");
            fprintf(stderr, "Input: 15-byte SensorPackets on stdin\n");
            fprintf(stderr, "Output: 200-byte RGBW frames (50×4) on stdout\n");
            fprintf(stderr, "--json: write per-frame diagnostics to stderr as JSONL\n");
            return 0;
        }
    }

    simInit(12345);
    simSetAlgorithm(alg);

    float dt = 1.0f / fps;
    uint32_t nowMs = 0;
    uint32_t frameIdx = 0;
    SensorPacket pkt;

    while (fread(&pkt, sizeof(SensorPacket), 1, stdin) == 1) {
        nowMs += (uint32_t)(dt * 1000.0f);
        simStep(pkt, dt, nowMs);
        const RgbwPixel* fb = simGetFramebuffer();

        // Write raw RGBW frame to stdout
        fwrite(fb, sizeof(RgbwPixel) * LED_COUNT, 1, stdout);

        if (jsonDiag) {
            // Compute frame stats
            uint32_t totalR = 0, totalG = 0, totalB = 0, totalW = 0;
            uint8_t maxR = 0, maxG = 0, maxB = 0, maxW = 0;
            int litPixels = 0;
            for (int i = 0; i < LED_COUNT; i++) {
                totalR += fb[i].r; totalG += fb[i].g;
                totalB += fb[i].b; totalW += fb[i].w;
                if (fb[i].r > maxR) maxR = fb[i].r;
                if (fb[i].g > maxG) maxG = fb[i].g;
                if (fb[i].b > maxB) maxB = fb[i].b;
                if (fb[i].w > maxW) maxW = fb[i].w;
                if (fb[i].r > 0 || fb[i].g > 0 || fb[i].b > 0 || fb[i].w > 0)
                    litPixels++;
            }
            fprintf(stderr,
                "{\"frame\":%u,\"t\":%.3f,\"lit\":%d,\"avgR\":%.1f,\"avgG\":%.1f,"
                "\"avgB\":%.1f,\"avgW\":%.1f,\"maxR\":%u,\"maxG\":%u,\"maxB\":%u,"
                "\"maxW\":%u,\"rms\":%d,\"mic\":%d}\n",
                frameIdx, nowMs / 1000.0f, litPixels,
                totalR / 50.0f, totalG / 50.0f, totalB / 50.0f, totalW / 50.0f,
                maxR, maxG, maxB, maxW,
                pkt.rawRms, pkt.micEnabled);
        }
        frameIdx++;
    }

    if (jsonDiag) {
        fprintf(stderr, "{\"summary\":{\"frames\":%u,\"effect\":\"%s\",\"fps\":%.1f}}\n",
                frameIdx, algName(alg), fps);
    }

    return 0;
}
