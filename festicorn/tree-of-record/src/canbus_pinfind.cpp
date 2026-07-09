// canbus_pinfind — find which GPIOs are wired to the CAN transceiver, in one
// flash. A transceiver loops TXD -> bus -> RXD combinationally, so driving the
// true TX pin low pulls the true RX pin low (dominant), and releasing it back
// high returns RX high (recessive). Sweeps all ordered candidate pairs.
// Other boards on the bus must be parked recessive (canbus_park) or the bus
// will read dominant regardless.
//
// DIAGNOSTIC ENV: reflash real firmware afterward.

#include <Arduino.h>

static const int CAND[] = {2, 4, 5, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33};
static const int N = sizeof(CAND)/sizeof(CAND[0]);
static const int RXONLY[] = {34, 35, 36, 39};   // input-only, no internal pulls
static const int NR = sizeof(RXONLY)/sizeof(RXONLY[0]);

void setup() {
  Serial.begin(115200);
  delay(300);
}

void loop() {
  Serial.println("\n[pinfind] idle levels with INPUT_PULLDOWN (HIGH = actively driven, likely RXD):");
  for (int i = 0; i < N; i++) {
    pinMode(CAND[i], INPUT_PULLDOWN);
  }
  delayMicroseconds(200);
  for (int i = 0; i < N; i++) {
    Serial.printf("  GPIO%-2d = %s\n", CAND[i], digitalRead(CAND[i]) ? "HIGH" : "low");
  }

  // Sweep with every non-RX pin driven HIGH: wherever this board's own TXD is,
  // it's held recessive, so a pulled-down pin can't jam the bus during the test.
  Serial.println("[pinfind] pair sweep (all pins parked HIGH, drive t low, watch r):");
  int hits = 0;
  for (int ti = 0; ti < N; ti++) {
    for (int ri = 0; ri < N; ri++) {
      if (ti == ri) continue;
      int t = CAND[ti], r = CAND[ri];
      for (int k = 0; k < N; k++) {
        if (CAND[k] == r) continue;
        pinMode(CAND[k], OUTPUT); digitalWrite(CAND[k], HIGH);
      }
      pinMode(r, INPUT_PULLDOWN);
      delayMicroseconds(100);
      int hi1 = digitalRead(r);
      digitalWrite(t, LOW);  delayMicroseconds(50);
      int lo  = digitalRead(r);
      digitalWrite(t, HIGH); delayMicroseconds(50);
      int hi2 = digitalRead(r);
      if (hi1 && !lo && hi2) {
        Serial.printf("  MATCH: TX=GPIO%d RX=GPIO%d\n", t, r);
        hits++;
      }
    }
  }
  Serial.println("[pinfind] input-only RX sweep (GPIO34/35/36/39, no pulls -- HIGH idle = likely RXD):");
  for (int k = 0; k < N; k++) { pinMode(CAND[k], OUTPUT); digitalWrite(CAND[k], HIGH); }
  for (int ri = 0; ri < NR; ri++) {
    pinMode(RXONLY[ri], INPUT);
    delayMicroseconds(100);
    Serial.printf("  GPIO%d idle=%s\n", RXONLY[ri], digitalRead(RXONLY[ri]) ? "HIGH" : "low");
    for (int ti = 0; ti < N; ti++) {
      int t = CAND[ti], r = RXONLY[ri];
      int hi1 = digitalRead(r);
      digitalWrite(t, LOW);  delayMicroseconds(50);
      int lo  = digitalRead(r);
      digitalWrite(t, HIGH); delayMicroseconds(50);
      int hi2 = digitalRead(r);
      if (hi1 && !lo && hi2) {
        Serial.printf("  MATCH: TX=GPIO%d RX=GPIO%d\n", t, r);
        hits++;
      }
    }
  }
  for (int k = 0; k < N; k++) { pinMode(CAND[k], OUTPUT); digitalWrite(CAND[k], HIGH); }  // leave bus recessive
  if (!hits) Serial.println("  no pair found among candidates");
  delay(3000);
}
