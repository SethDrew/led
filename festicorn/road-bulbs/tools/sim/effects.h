#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

#define DESKTOP_SIM 1
#define LED_COUNT 50

struct __attribute__((packed)) SensorPacket {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    uint16_t rawRms;
    uint8_t micEnabled;
};  // 15 bytes

enum Algorithm {
    ALG_SPARKLE_BURST,
    ALG_FIRE_MELD,
    ALG_FIRE_FLICKER,
    ALG_QUIET_BLOOM,
    ALG_GRAVITY_PARTICLE,
    ALG_SPARKLE_SYLLABLE,
};

struct RgbwPixel { uint8_t r, g, b, w; };

void simInit(uint32_t seed);
void simSetAlgorithm(Algorithm alg);
void simStep(const SensorPacket &pkt, float dt, uint32_t nowMs);
const RgbwPixel* simGetFramebuffer();  // returns LED_COUNT pixels
Algorithm simGetAlgorithm();
void simInjectOnset(uint8_t strength, uint8_t band);
