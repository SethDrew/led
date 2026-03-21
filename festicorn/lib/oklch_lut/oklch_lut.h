#pragma once
#include <stdint.h>

// OKLCH constant-L rainbow LUT (L=0.75, per-hue max chroma, 256 entries)
extern const uint8_t oklchConstL[256][3];

// OKLCH variable-L rainbow LUT (hue-dependent lightness, 256 entries)
// Red (~30°) L≈0.52, purple (~300°) L≈0.38, green/cyan/yellow L=0.75
extern const uint8_t oklchVarL[256][3];
