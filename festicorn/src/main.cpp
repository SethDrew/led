#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoOTA.h>
#include <ESPmDNS.h>
#include <LittleFS.h>
#include <Adafruit_NeoPixel.h>
#include "effects.h"
#include "web_ui.h"
#include "wifi_credentials_local.h"

// All output pins
static const uint8_t PINS[] = {1, 5, 6, 7, 8, 9, 10, 20, 21};
static const uint8_t NUM_STRIPS = sizeof(PINS) / sizeof(PINS[0]);
Adafruit_NeoPixel strips[NUM_STRIPS];
Adafruit_NeoPixel &strip = strips[0]; // web API controls brightness via first strip
WebServer server(80);
EffectState state;

// --- File upload state ---
File uploadFile;
bool uploadOk = false;

// --- Web handlers ---

void handleRoot() {
    File f = LittleFS.open("/index.html", "r");
    if (f) {
        server.streamFile(f, "text/html");
        f.close();
    } else {
        // No UI on flash yet — serve PROGMEM fallback (which is the upload page)
        server.send(200, "text/html", UPLOAD_PAGE);
    }
}

void handleUploadPage() {
    server.send(200, "text/html", UPLOAD_PAGE);
}

void handleSource() {
    File f = LittleFS.open("/index.html", "r");
    if (f) {
        server.streamFile(f, "text/plain");
        f.close();
    } else {
        server.send(404, "text/plain", "No UI uploaded yet");
    }
}

void handleUploadComplete() {
    if (uploadOk) {
        server.sendHeader("Location", "/upload?ok=1");
    } else {
        server.sendHeader("Location", "/upload?err=write+failed");
    }
    server.send(303);
}

void handleUploadData() {
    HTTPUpload& upload = server.upload();
    if (upload.status == UPLOAD_FILE_START) {
        Serial.printf("[Upload] Start: %s\n", upload.filename.c_str());
        uploadFile = LittleFS.open("/index.html", "w");
        uploadOk = false;
    } else if (upload.status == UPLOAD_FILE_WRITE) {
        if (uploadFile) {
            uploadFile.write(upload.buf, upload.currentSize);
        }
    } else if (upload.status == UPLOAD_FILE_END) {
        if (uploadFile) {
            uploadFile.close();
            uploadOk = true;
            Serial.printf("[Upload] Done: %u bytes\n", upload.totalSize);
        }
    }
}

void handleStatus() {
    String json = "{";
    json += "\"effect\":\"" + String(state.effect == RAINBOW ? "rainbow" : "gradient") + "\",";
    json += "\"brightness\":" + String(state.brightness) + ",";
    json += "\"cycleTime\":" + String(state.cycleTimeMs) + ",";
    const char *palName;
    switch (state.palette) {
        case SAP_FLOW:         palName = "sap_flow";         break;
        case OKLCH_RAINBOW:    palName = "oklch_rainbow";    break;
        // Category 2: Hue-arc gradients
        case RED_BLUE:         palName = "red_blue";         break;
        case CYAN_GOLD:        palName = "cyan_gold";        break;
        case GREEN_PURPLE:     palName = "green_purple";     break;
        case ORANGE_TEAL:      palName = "orange_teal";      break;
        case MAGENTA_CYAN:     palName = "magenta_cyan";     break;
        case SUNSET_SKY:       palName = "sunset_sky";       break;
        // Chroma sweeps
        case BLUE_WASH:        palName = "blue_wash";        break;
        case RED_WASH:         palName = "red_wash";         break;
        case GREEN_WASH:       palName = "green_wash";       break;
        case PURPLE_WASH:      palName = "purple_wash";      break;
        case GOLD_WASH:        palName = "gold_wash";        break;
        default:               palName = "sap_flow";         break;
    }
    json += "\"palette\":\"" + String(palName) + "\",";
    json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
    json += "\"ip\":\"" + WiFi.localIP().toString() + "\"";
    json += "}";
    server.send(200, "application/json", json);
}

void handleSetEffect() {
    if (server.hasArg("effect")) {
        String e = server.arg("effect");
        if (e == "rainbow") state.effect = RAINBOW;
        else if (e == "gradient") state.effect = GRADIENT;
    }
    server.send(200, "text/plain", "OK");
}

void handleSetBrightness() {
    if (server.hasArg("brightness")) {
        int b = server.arg("brightness").toInt();
        if (b >= 0 && b <= 255) state.brightness = b;
    }
    server.send(200, "text/plain", "OK");
}

void handleSetCycleTime() {
    if (server.hasArg("cycletime")) {
        long ct = server.arg("cycletime").toInt();
        if (ct >= 1000 && ct <= 600000) state.cycleTimeMs = ct;
    }
    server.send(200, "text/plain", "OK");
}

void handleSetPalette() {
    if (server.hasArg("palette")) {
        String p = server.arg("palette");
        if (p == "sap_flow")              state.palette = SAP_FLOW;
        else if (p == "oklch_rainbow")    state.palette = OKLCH_RAINBOW;
        // Category 2: Hue-arc gradients
        else if (p == "red_blue")         state.palette = RED_BLUE;
        else if (p == "cyan_gold")        state.palette = CYAN_GOLD;
        else if (p == "green_purple")     state.palette = GREEN_PURPLE;
        else if (p == "orange_teal")      state.palette = ORANGE_TEAL;
        else if (p == "magenta_cyan")     state.palette = MAGENTA_CYAN;
        else if (p == "sunset_sky")       state.palette = SUNSET_SKY;
        // Chroma sweeps
        else if (p == "blue_wash")        state.palette = BLUE_WASH;
        else if (p == "red_wash")         state.palette = RED_WASH;
        else if (p == "green_wash")       state.palette = GREEN_WASH;
        else if (p == "purple_wash")      state.palette = PURPLE_WASH;
        else if (p == "gold_wash")        state.palette = GOLD_WASH;
    }
    server.send(200, "text/plain", "OK");
}

// --- Setup ---

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n[Butterfly] Starting...");

    // LittleFS
    if (!LittleFS.begin(true)) {
        Serial.println("[FS] LittleFS mount failed");
    } else {
        Serial.println("[FS] LittleFS mounted");
    }

    // LED strips — all pins
    for (uint8_t i = 0; i < NUM_STRIPS; i++) {
        strips[i].updateType(LED_TYPE);
        strips[i].updateLength(NUM_PIXELS);
        strips[i].setPin(PINS[i]);
        strips[i].begin();
        strips[i].setBrightness(state.brightness);
        strips[i].show();
        Serial.printf("  Strip %d: pin %d\n", i, PINS[i]);
    }

    // WiFi
    WiFi.mode(WIFI_STA);
    WiFi.setHostname("butterfly");

    // Scan to see what's visible
    Serial.println("[WiFi] Scanning...");
    int n = WiFi.scanNetworks();
    for (int i = 0; i < n; i++) {
        Serial.printf("  %s (%ddBm)\n", WiFi.SSID(i).c_str(), WiFi.RSSI(i));
    }
    if (n == 0) Serial.println("  No networks found!");

    // Print scan details (auth type helps diagnose)
    for (int i = 0; i < n; i++) {
        if (WiFi.SSID(i) == WIFI_SSID) {
            Serial.printf("  -> auth type: %d  channel: %d\n", WiFi.encryptionType(i), WiFi.channel(i));
        }
    }

    // Try relaxed security settings for WPA2/WPA3 transition routers
    WiFi.setMinSecurity(WIFI_AUTH_WPA_PSK);

    WiFi.setAutoReconnect(true);
    WiFi.persistent(false);

    Serial.printf("[WiFi] Connecting to '%s'...", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD, 0, NULL, true);
    unsigned long wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 20000) {
        delay(500);
        Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.printf("\n[WiFi] FAILED (status=%d) — LEDs running without network\n", WiFi.status());
        // Retry once
        Serial.println("[WiFi] Retrying...");
        WiFi.disconnect();
        delay(1000);
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD, 0, NULL, true);
        wifiStart = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 20000) {
            delay(500);
            Serial.print(".");
        }
        if (WiFi.status() == WL_CONNECTED) {
            Serial.printf("\n[WiFi] Connected on retry: %s\n", WiFi.localIP().toString().c_str());
        } else {
            Serial.printf("\n[WiFi] FAILED again (status=%d)\n", WiFi.status());
        }
    }

    if (WiFi.status() == WL_CONNECTED) {
        MDNS.begin("butterfly");
    }

    // OTA
    ArduinoOTA.setHostname("butterfly");
    ArduinoOTA.onStart([]() { Serial.println("[OTA] Start"); });
    ArduinoOTA.onEnd([]() { Serial.println("\n[OTA] Done"); });
    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
        Serial.printf("[OTA] %u%%\r", progress * 100 / total);
    });
    ArduinoOTA.onError([](ota_error_t error) {
        Serial.printf("[OTA] Error %u\n", error);
    });
    ArduinoOTA.begin();

    // Web server
    server.on("/", HTTP_GET, handleRoot);
    server.on("/source", HTTP_GET, handleSource);
    server.on("/upload", HTTP_GET, handleUploadPage);
    server.on("/upload", HTTP_POST, handleUploadComplete, handleUploadData);
    server.on("/status", HTTP_GET, handleStatus);
    server.on("/effect", HTTP_POST, handleSetEffect);
    server.on("/brightness", HTTP_POST, handleSetBrightness);
    server.on("/cycletime", HTTP_POST, handleSetCycleTime);
    server.on("/palette", HTTP_POST, handleSetPalette);
    server.begin();
    Serial.println("[Web] Server started on port 80");
}

// --- Main loop ---

void loop() {
    // WiFi reconnect
    if (WiFi.status() != WL_CONNECTED) {
        static unsigned long lastReconnect = 0;
        if (millis() - lastReconnect > 30000) {
            Serial.println("[WiFi] Reconnecting...");
            WiFi.disconnect();
            WiFi.begin(WIFI_SSID, WIFI_PASSWORD, 0, NULL, true);
            lastReconnect = millis();
        }
    }

    ArduinoOTA.handle();
    server.handleClient();

    for (uint8_t i = 0; i < NUM_STRIPS; i++) {
        // Brightness is applied via gammaHybrid inside render functions
        // (gamma on scalar brightness, not per-channel). Don't use
        // NeoPixel setBrightness() which would double-dim.
        strips[i].setBrightness(255);

        switch (state.effect) {
            case GRADIENT:
                renderGradient(strips[i], state);
                break;
            default: // RAINBOW
                renderRainbow(strips[i], state);
                break;
        }

        strips[i].show();
    }

    delay(16); // ~60fps
}
