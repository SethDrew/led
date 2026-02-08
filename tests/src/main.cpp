/*
 * Comprehensive LED Strip Diagnostic
 * Tests: Arduino working, pin output, different LED protocols
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 12
#define ONBOARD_LED 13

Adafruit_NeoPixel stripGRB(10, LED_PIN, NEO_GRB + NEO_KHZ800);  // WS2812B
Adafruit_NeoPixel stripRGB(10, LED_PIN, NEO_RGB + NEO_KHZ800);  // Alternate

int testPhase = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== LED STRIP DIAGNOSTIC ===");
  Serial.println("Testing pin 12");

  pinMode(LED_PIN, OUTPUT);
  pinMode(ONBOARD_LED, OUTPUT);

  stripGRB.begin();
  stripGRB.setBrightness(255);
  stripRGB.begin();
  stripRGB.setBrightness(255);
}

void loop() {
  Serial.print("\nTest ");
  Serial.print(testPhase);
  Serial.print(": ");

  switch(testPhase) {
    case 0:
      Serial.println("Arduino alive check - onboard LED blink");
      for(int i = 0; i < 5; i++) {
        digitalWrite(ONBOARD_LED, HIGH);
        delay(200);
        digitalWrite(ONBOARD_LED, LOW);
        delay(200);
      }
      break;

    case 1:
      Serial.println("Pin 12 toggle test - measure with multimeter");
      Serial.println("Should see ~2.5V average if working");
      for(int i = 0; i < 50; i++) {
        digitalWrite(LED_PIN, HIGH);
        delayMicroseconds(100);
        digitalWrite(LED_PIN, LOW);
        delayMicroseconds(100);
      }
      delay(2000);
      break;

    case 2:
      Serial.println("WS2812B (GRB) - First LED RED");
      stripGRB.clear();
      stripGRB.setPixelColor(0, stripGRB.Color(255, 0, 0));
      stripGRB.show();
      delay(2000);
      break;

    case 3:
      Serial.println("WS2812B (GRB) - First LED GREEN");
      stripGRB.clear();
      stripGRB.setPixelColor(0, stripGRB.Color(0, 255, 0));
      stripGRB.show();
      delay(2000);
      break;

    case 4:
      Serial.println("WS2812B (GRB) - First LED BLUE");
      stripGRB.clear();
      stripGRB.setPixelColor(0, stripGRB.Color(0, 0, 255));
      stripGRB.show();
      delay(2000);
      break;

    case 5:
      Serial.println("WS2812B (GRB) - First LED WHITE (full brightness)");
      stripGRB.clear();
      stripGRB.setPixelColor(0, stripGRB.Color(255, 255, 255));
      stripGRB.show();
      delay(2000);
      break;

    case 6:
      Serial.println("Alternate RGB order - First LED RED");
      stripRGB.clear();
      stripRGB.setPixelColor(0, stripRGB.Color(255, 0, 0));
      stripRGB.show();
      delay(2000);
      break;

    case 7:
      Serial.println("ALL 10 LEDs WHITE (in case first LED is dead)");
      stripGRB.clear();
      for(int i = 0; i < 10; i++) {
        stripGRB.setPixelColor(i, stripGRB.Color(255, 255, 255));
      }
      stripGRB.show();
      delay(3000);
      break;
  }

  testPhase++;
  if(testPhase > 7) {
    Serial.println("\n=== Cycle complete ===");
    Serial.println("Did you see:");
    Serial.println("- Onboard LED blink? (Arduino working)");
    Serial.println("- Any strip LEDs light? (which test?)");
    Serial.println("- Multimeter show voltage? (pin working)");
    Serial.println("\nRestarting tests...\n");
    testPhase = 0;
    delay(3000);
  }

  delay(500);
}
