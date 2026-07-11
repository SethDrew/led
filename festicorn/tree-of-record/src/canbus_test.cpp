// canbus_test — minimal TWAI two-node bring-up. Same binary on both boards;
// node ID is the low MAC byte, so each board labels its own frames. Each node
// heartbeats every 500ms and prints every frame it receives, plus periodic bus
// health (a climbing tx_error_counter / bus-off state = no ACK = wiring fault).
//
// DIAGNOSTIC ENV: reflash the board's real firmware (env:tree-of-record-eth /
// env:streaming-eth) after this test.
//
// Wiring per node: transceiver required at EACH node, common ground, 120Ω
// termination at both bus ends. TX=GPIO16, RX=GPIO17 (clean GPIOs, no strapping
// concerns; TWAI routes either pin via the GPIO matrix).

#include <Arduino.h>
#include "driver/twai.h"
#include "esp_mac.h"

#ifndef CAN_TX_PIN
#define CAN_TX_PIN 16
#endif
#ifndef CAN_RX_PIN
#define CAN_RX_PIN 17
#endif
static const gpio_num_t TX_PIN = (gpio_num_t)CAN_TX_PIN;
static const gpio_num_t RX_PIN = (gpio_num_t)CAN_RX_PIN;

static uint8_t  nodeId  = 0;
static uint32_t txCount = 0;

static void fatal(const char* what) {
  Serial.printf("[canbus] %s FAILED\n", what);
  while (true) delay(1000);
}

void setup() {
  Serial.begin(115200);
  delay(300);

  uint8_t mac[6];
  esp_read_mac(mac, ESP_MAC_WIFI_STA);
  nodeId = mac[5];
  Serial.printf("\n[canbus] node=0x%02X  TX=GPIO%d RX=GPIO%d @500kbps\n",
                nodeId, (int)TX_PIN, (int)RX_PIN);

#ifdef CAN_SELFTEST
  twai_mode_t mode = TWAI_MODE_NO_ACK;   // self-test: no second node needed to ACK
#else
  twai_mode_t mode = TWAI_MODE_NORMAL;
#endif
  twai_general_config_t g = TWAI_GENERAL_CONFIG_DEFAULT(TX_PIN, RX_PIN, mode);
  twai_timing_config_t  t = TWAI_TIMING_CONFIG_500KBITS();
  twai_filter_config_t  f = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  if (twai_driver_install(&g, &t, &f) != ESP_OK) fatal("driver install");
  if (twai_start() != ESP_OK) fatal("start");
  Serial.printf("[canbus] driver started (%s), listening + heartbeating\n",
                mode == TWAI_MODE_NO_ACK ? "SELF-TEST/no-ack" : "normal");
}

void loop() {
  const uint32_t now = millis();

  static uint32_t lastTx = 0;
  if (now - lastTx >= 500) {
    lastTx = now;
    // Political discourse, 8 bytes per hot take. Node ID staggers the
    // rotation so the two boards argue instead of agreeing.
    static const char* TAKES[] = {
      "TAX RGB!", "NO TAXES", "VOTE LED", "VOTE CAN", "FREE HUE",
      "GAMMA 4A", "PWM 4EVA", "DIM LEFT", "BRT RGHT", "ACK ME!!",
    };
    static const int NTAKES = sizeof(TAKES) / sizeof(TAKES[0]);
    twai_message_t msg = {};
    msg.identifier       = 0x100 | nodeId;   // 11-bit standard, unique per node
    msg.data_length_code = 8;
    memcpy(msg.data, TAKES[(txCount + nodeId) % NTAKES], 8);
#ifdef CAN_SELFTEST
    msg.self = 1;   // hear our own frame back through the transceiver
#endif
    esp_err_t r = twai_transmit(&msg, pdMS_TO_TICKS(100));
    Serial.printf("[TX] id=0x%03X n=%lu -> %s\n",
                  (unsigned)msg.identifier, (unsigned long)txCount,
                  r == ESP_OK ? "queued" : esp_err_to_name(r));
    txCount++;
  }

  twai_message_t rx;
  while (twai_receive(&rx, 0) == ESP_OK) {
    Serial.printf("[RX] id=0x%03X dlc=%u data=", (unsigned)rx.identifier,
                  rx.data_length_code);
    for (int i = 0; i < rx.data_length_code; i++) Serial.printf("%02X ", rx.data[i]);
    Serial.printf(" \"%.8s\"\n", (const char*)rx.data);
  }

  static uint32_t lastStat = 0;
  if (now - lastStat >= 2000) {
    lastStat = now;
    twai_status_info_t st;
    twai_get_status_info(&st);
    const char* state = st.state == TWAI_STATE_RUNNING ? "RUNNING"
                      : st.state == TWAI_STATE_BUS_OFF ? "BUS_OFF"
                      : st.state == TWAI_STATE_RECOVERING ? "RECOVERING" : "STOPPED";
    Serial.printf("[STAT] state=%s tx_err=%u rx_err=%u tx_fail=%u rx_miss=%u bus_err=%u arb_lost=%u\n",
                  state, st.tx_error_counter, st.rx_error_counter,
                  st.tx_failed_count, st.rx_missed_count, st.bus_error_count,
                  st.arb_lost_count);
    if (st.state == TWAI_STATE_BUS_OFF) {
      Serial.println("[canbus] BUS_OFF — initiating recovery");
      twai_initiate_recovery();
    }
  }

  delay(5);
}
