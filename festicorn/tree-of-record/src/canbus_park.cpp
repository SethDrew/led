// canbus_park — drive all candidate CAN TX pins HIGH so this board's
// transceiver holds the bus recessive while another board is under test.
// (A floating TXD can jam the bus dominant; holding boards in reset is NOT safe.)
//
// DIAGNOSTIC ENV: reflash real firmware afterward.

#include <Arduino.h>

static const int PINS[] = {2, 4, 16, 17};

void setup() {
  Serial.begin(115200);
  for (int p : PINS) { pinMode(p, OUTPUT); digitalWrite(p, HIGH); }
  Serial.println("\n[park] pins 2/4/16/17 driven HIGH (bus recessive)");
}

void loop() { delay(1000); }
