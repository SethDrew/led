#pragma once
#include <stdint.h>

static const uint32_t RFB_MAGIC = 0x31424652;
static const uint8_t CHANNEL = 1;
static const int MAX_PEERS = 8;

enum : uint8_t { PKT_PING = 1, PKT_PONG = 2, PKT_FLOOD = 3, PKT_REPORT = 4 };

enum : uint8_t { RFCHIP_ESP32 = 0, RFCHIP_C3 = 1 };

struct __attribute__((packed)) RfHdr {
    uint32_t magic;
    uint8_t type;
    int8_t txq;
    uint16_t seq;
    uint32_t t_send;
};

struct __attribute__((packed)) PeerRpt {
    uint8_t mac[6];
    uint32_t count;
    int32_t rssi_sum;
    int32_t nf_sum;
    int8_t rssi_min;
    int8_t rssi_max;
};

struct __attribute__((packed)) NodeReport {
    uint8_t chip;
    int8_t txq_readback;
    uint32_t uptime_s;
    uint32_t window;
    uint16_t window_ms;
    uint8_t n_peers;
    PeerRpt peer[4];
};
