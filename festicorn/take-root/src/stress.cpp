// =============================================================================
// TEMPORARY STRESS TEST — DELETE AFTER BENCH VERIFICATION
// =============================================================================
// Purpose: hammer the ESP32 radio so the user can probe the buck-converter 5V
// rail under realistic worst-case TX load. LEDs are NOT driven here (they run
// from a separate 12V supply, not the buck under test).
//
// Behaviour:
//   1. Connect to "cuteplant" as a STA.
//   2. Loop WiFi.scanNetworks() with no inter-scan delay. Each scan ≈ 2s of
//      sustained active radio bursts.
//   3. Print millis(), free heap, scan count, and first-hit RSSI to serial so
//      the user can confirm it's running.
//
// Cleanup, when done:
//   - Delete this file.
//   - Delete the [env:take-root-stress] block in platformio.ini.
//   - Leave main.cpp and [env:take-root] untouched.
// =============================================================================

#include <Arduino.h>
#include <WiFi.h>

// Inline credentials — matches the convention used in
// festicorn/bench-tree/src/tree_rainbow_test.cpp. Acceptable here because this
// file is temporary and will be deleted after bench verification.
#define WIFI_SSID     "cuteplant"
#define WIFI_PASSWORD "bigboiredwood"

static uint32_t g_scan_iter = 0;

static void connectWifi() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.printf("[wifi] connecting to '%s'", WIFI_SSID);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
        delay(250);
        Serial.print('.');
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[wifi] connected, IP=%s, RSSI=%d dBm, ch=%d\n",
                      WiFi.localIP().toString().c_str(),
                      WiFi.RSSI(),
                      WiFi.channel());
    } else {
        Serial.println("\n[wifi] connect timed out — proceeding to scan loop anyway");
    }
}

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println();
    Serial.println("=== take-root-stress: WiFi radio stress test ===");
    Serial.println("LEDs are NOT driven. Probe 5V rail at the buck output.");
    connectWifi();
}

void loop() {
    int n = WiFi.scanNetworks(/*async=*/false, /*show_hidden=*/true);
    g_scan_iter++;

    int rssi0 = 0;
    const char* ssid0 = "(none)";
    if (n > 0) {
        rssi0 = WiFi.RSSI(0);
        ssid0 = WiFi.SSID(0).c_str();
    }
    Serial.printf("[scan %lu] t=%lums heap=%u found=%d first='%s' rssi=%d link=%d\n",
                  (unsigned long)g_scan_iter,
                  (unsigned long)millis(),
                  (unsigned)ESP.getFreeHeap(),
                  n,
                  ssid0,
                  rssi0,
                  (int)WiFi.status());

    WiFi.scanDelete();
    // No delay — back-to-back scans for sustained TX activity.
}
