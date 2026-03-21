/*
 * DIFFUSER STICK — STANDALONE SAP FLOW
 *
 * Ported from tree SapFlowForeground. Particles rise along
 * the 23-LED stick on a deep green background.
 * Runs entirely on the ESP32 — no streaming required.
 */

#include <Adafruit_NeoPixel.h>

#define TOTAL_LEDS (STRIP_A_COUNT + STRIP_B_COUNT)

Adafruit_NeoPixel stripA(STRIP_A_COUNT, STRIP_A_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripB(STRIP_B_COUNT, STRIP_B_PIN, NEO_GRB + NEO_KHZ800);

// Frame buffer: R, G, B per LED
uint8_t leds[TOTAL_LEDS][3];

// --- Sap flow parameters (from tree SapFlowForeground.h) ---

struct SapParticle {
    float pos;
    float velocity;
    uint8_t brightness;
    bool active;
};

static const uint8_t MAX_PARTICLES = 12;
SapParticle particles[MAX_PARTICLES];

// Particle color: bright lime green
static const uint8_t SAP_R = 100;
static const uint8_t SAP_G = 255;
static const uint8_t SAP_B = 60;

// Background: deep forest green (visible but subtle)
static const uint8_t BG_R = 1;
static const uint8_t BG_G = 6;
static const uint8_t BG_B = 1;

// Spawn timing (tuned for 0.3 velocity)
static const uint8_t SPAWN_CHANCE = 5;         // % per eligible frame
static const uint8_t MIN_FRAMES_BETWEEN = 27;
static const uint8_t MAX_FRAMES_BETWEEN = 160;

// Rendering
static const float VELOCITY = 0.3;
static const float SOFT_RADIUS = 3.0;

uint8_t framesSinceLastSpawn = 0;

// Target ~60 fps
static const unsigned long FRAME_MS = 16;

void showFrame() {
    for (uint16_t i = 0; i < STRIP_A_COUNT; i++)
        stripA.setPixelColor(i, leds[i][0], leds[i][1], leds[i][2]);
    for (uint16_t i = 0; i < STRIP_B_COUNT; i++)
        stripB.setPixelColor(i, leds[STRIP_A_COUNT + i][0], leds[STRIP_A_COUNT + i][1], leds[STRIP_A_COUNT + i][2]);
    stripA.show();
    stripB.show();
}

void setup() {
    Serial.begin(115200);

    stripA.begin();
    stripA.setBrightness(255);
    stripB.begin();
    stripB.setBrightness(255);

    // Startup flash: brief red then off
    for (uint16_t i = 0; i < STRIP_A_COUNT; i++) stripA.setPixelColor(i, 50, 0, 0);
    for (uint16_t i = 0; i < STRIP_B_COUNT; i++) stripB.setPixelColor(i, 50, 0, 0);
    stripA.show();
    stripB.show();
    delay(200);
    stripA.clear(); stripA.show();
    stripB.clear(); stripB.show();

    for (uint8_t i = 0; i < MAX_PARTICLES; i++)
        particles[i].active = false;

    Serial.println("Diffuser Stick — Sap Flow");
    Serial.printf("Strip A: pin %d, %d LEDs\n", STRIP_A_PIN, STRIP_A_COUNT);
    Serial.printf("Strip B: pin %d, %d LEDs\n", STRIP_B_PIN, STRIP_B_COUNT);
}

void loop() {
    unsigned long frameStart = millis();

    framesSinceLastSpawn++;

    // --- Spawn ---
    bool shouldSpawn = false;
    if (framesSinceLastSpawn >= MAX_FRAMES_BETWEEN) {
        shouldSpawn = true;
    } else if (framesSinceLastSpawn >= MIN_FRAMES_BETWEEN && random(100) < SPAWN_CHANCE) {
        shouldSpawn = true;
    }

    if (shouldSpawn) {
        for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
            if (!particles[i].active) {
                particles[i].active = true;
                particles[i].pos = 0;
                particles[i].velocity = VELOCITY;
                particles[i].brightness = 80 + random(70);
                framesSinceLastSpawn = 0;
                break;
            }
        }
    }

    // --- Update ---
    for (uint8_t i = 0; i < MAX_PARTICLES; i++) {
        if (!particles[i].active) continue;
        particles[i].pos += particles[i].velocity;
        if (particles[i].pos > TOTAL_LEDS + 5)
            particles[i].active = false;
    }

    // --- Render ---
    // Fill background
    for (uint8_t i = 0; i < TOTAL_LEDS; i++) {
        leds[i][0] = BG_R;
        leds[i][1] = BG_G;
        leds[i][2] = BG_B;
    }

    // Additive particle rendering with soft edges
    for (uint8_t p = 0; p < MAX_PARTICLES; p++) {
        if (!particles[p].active) continue;

        float particlePos = particles[p].pos;
        uint8_t brightness = particles[p].brightness;

        for (uint8_t i = 0; i < TOTAL_LEDS; i++) {
            float distance = fabs((float)i - particlePos);
            if (distance < SOFT_RADIUS) {
                float falloff = 1.0f - (distance / SOFT_RADIUS);
                falloff = falloff * falloff;

                uint8_t pr = (uint16_t)SAP_R * brightness * falloff / 255;
                uint8_t pg = (uint16_t)SAP_G * brightness * falloff / 255;
                uint8_t pb = (uint16_t)SAP_B * brightness * falloff / 255;

                leds[i][0] = min(255, leds[i][0] + pr);
                leds[i][1] = min(255, leds[i][1] + pg);
                leds[i][2] = min(255, leds[i][2] + pb);
            }
        }
    }

    showFrame();

    // Hold frame rate
    unsigned long elapsed = millis() - frameStart;
    if (elapsed < FRAME_MS)
        delay(FRAME_MS - elapsed);
}
