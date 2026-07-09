// canbus_throughput — max-rate profiler for TWAI on classic ESP32.
// One node floods max-size frames, the other counts delivered frames/bytes.
// Ground truth = what the SINK actually receives (the flooder's queued count is
// only an upper bound). Serial can't keep up with per-frame prints at line rate,
// so we only print a 1 Hz summary.
//
// Build flags:
//   -DROLE_FLOOD | -DROLE_SINK   which side this board plays
//   -DCAN_KBPS=1000|500|250|125  bitrate
//   -DTX_PATTERN=0|1|2           0=counter, 1=0x55/0xAA (min bit-stuffing),
//                                2=0x00 (max bit-stuffing)
//   -DTX_DLC=8                   payload bytes per frame (0..8)
//
// DIAGNOSTIC ENV: reflash real firmware afterward.

#include <Arduino.h>
#include "driver/twai.h"
#include "esp_mac.h"

#ifndef CAN_KBPS
#define CAN_KBPS 1000
#endif
#ifndef TX_PATTERN
#define TX_PATTERN 1
#endif
#ifndef TX_DLC
#define TX_DLC 8
#endif

static const gpio_num_t TX_PIN = GPIO_NUM_16;
static const gpio_num_t RX_PIN = GPIO_NUM_17;

static twai_timing_config_t timing() {
#if   CAN_KBPS == 1000
  return (twai_timing_config_t)TWAI_TIMING_CONFIG_1MBITS();
#elif CAN_KBPS == 500
  return (twai_timing_config_t)TWAI_TIMING_CONFIG_500KBITS();
#elif CAN_KBPS == 250
  return (twai_timing_config_t)TWAI_TIMING_CONFIG_250KBITS();
#else
  return (twai_timing_config_t)TWAI_TIMING_CONFIG_125KBITS();
#endif
}

void setup() {
  Serial.begin(115200);
  delay(300);
  uint8_t mac[6]; esp_read_mac(mac, ESP_MAC_WIFI_STA);

  twai_general_config_t g = TWAI_GENERAL_CONFIG_DEFAULT(TX_PIN, RX_PIN, TWAI_MODE_NORMAL);
  g.tx_queue_len = 32;
  g.rx_queue_len = 128;
  twai_timing_config_t t = timing();
  twai_filter_config_t f = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  if (twai_driver_install(&g, &t, &f) != ESP_OK || twai_start() != ESP_OK) {
    Serial.println("[thru] driver FAILED"); while (true) delay(1000);
  }
#if defined(ROLE_FLOOD)
  const char* role = "FLOOD";
#else
  const char* role = "SINK";
#endif
  Serial.printf("\n[thru] node=0x%02X role=%s bitrate=%dk dlc=%d pattern=%d\n",
                mac[5], role, (int)CAN_KBPS, (int)TX_DLC, (int)TX_PATTERN);
}

void loop() {
#if defined(ROLE_FLOOD)
  static twai_message_t msg = []{
    twai_message_t m = {};
    m.identifier = 0x123;
    m.data_length_code = TX_DLC;
    for (int i = 0; i < TX_DLC; i++)
#if   TX_PATTERN == 1
      m.data[i] = (i & 1) ? 0xAA : 0x55;
#elif TX_PATTERN == 2
      m.data[i] = 0x00;
#else
      m.data[i] = i;
#endif
    return m;
  }();

  static uint32_t okc = 0, failc = 0, last = 0;
#if TX_PATTERN == 0
  static uint32_t ctr = 0;
  for (int i = 0; i < TX_DLC && i < 4; i++) msg.data[i] = (ctr >> (8 * i)) & 0xFF;
  ctr++;
#endif
  esp_err_t r = twai_transmit(&msg, pdMS_TO_TICKS(50));
  if (r == ESP_OK) okc++; else failc++;

  uint32_t now = millis();
  if (now - last >= 1000) {
    twai_status_info_t st; twai_get_status_info(&st);
    Serial.printf("[FLOOD] queued=%lu/s fail=%lu tx_err=%u state=%d\n",
                  (unsigned long)okc, (unsigned long)failc, st.tx_error_counter, st.state);
    if (st.state == TWAI_STATE_BUS_OFF) { twai_initiate_recovery(); }
    else if (st.state == TWAI_STATE_STOPPED) { twai_start(); }
    okc = failc = 0; last = now;
  }
#else  // ROLE_SINK
  static uint32_t frames = 0, bytes = 0, last = 0, missBase = 0;
  twai_message_t rx;
  for (int i = 0; i < 256; i++) {              // bounded batch, then fall through to timer
    if (twai_receive(&rx, 0) != ESP_OK) break;
    frames++; bytes += rx.data_length_code;
  }
  uint32_t now = millis();
  if (now - last >= 1000) {
    twai_status_info_t st; twai_get_status_info(&st);
    uint32_t missed = st.rx_missed_count - missBase; missBase = st.rx_missed_count;
    float payload_kbps = (bytes * 8.0f) / 1000.0f;
    Serial.printf("[SINK] frames/s=%lu payload=%.1f kbit/s (%.1f kB/s) missed=%lu rx_err=%u bus_err=%u\n",
                  (unsigned long)frames, payload_kbps, bytes / 1000.0f,
                  (unsigned long)missed, st.rx_error_counter, st.bus_error_count);
    frames = bytes = 0; last = now;
  }
#endif
}
