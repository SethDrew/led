#include <Arduino.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <ElegantOTA.h>
#include <ESPmDNS.h>
#include <LittleFS.h>
#include <Adafruit_NeoPixel.h>
#include "effects.h"
#include "web_ui.h"
#include "wifi_credentials_local.h"

// All output pins
static const uint8_t PINS[] = {5};
static const uint8_t NUM_STRIPS = sizeof(PINS) / sizeof(PINS[0]);
Adafruit_NeoPixel strips[NUM_STRIPS];
Adafruit_NeoPixel &strip = strips[0]; // web API controls brightness via first strip
AsyncWebServer server(80);
EffectState state;

// Cache index.html in RAM to avoid LittleFS blocking the render loop
// (ESP32-C3 is single-core — flash reads stall everything)
String cachedIndexHtml;

// --- Web handlers (async) ---

void handleRoot(AsyncWebServerRequest *request) {
    if (cachedIndexHtml.length() > 0) {
        request->send(200, "text/html", cachedIndexHtml);
    } else {
        request->send(200, "text/html", UPLOAD_PAGE);
    }
}

void handleUploadPage(AsyncWebServerRequest *request) {
    request->send(200, "text/html", UPLOAD_PAGE);
}

void handleSource(AsyncWebServerRequest *request) {
    File f = LittleFS.open("/index.html", "r");
    if (f) {
        String content = f.readString();
        f.close();
        request->send(200, "text/plain", content);
    } else {
        request->send(404, "text/plain", "No UI uploaded yet");
    }
}

// --- File upload (async) ---
static File uploadFile;
static bool uploadOk = false;

void handleUploadComplete(AsyncWebServerRequest *request) {
    if (uploadOk) {
        request->redirect("/upload?ok=1");
    } else {
        request->redirect("/upload?err=write+failed");
    }
}

void handleUploadData(AsyncWebServerRequest *request, const String& filename,
                      size_t index, uint8_t *data, size_t len, bool final) {
    if (index == 0) {
        Serial.printf("[Upload] Start: %s\n", filename.c_str());
        uploadFile = LittleFS.open("/index.html", "w");
        uploadOk = false;
    }
    if (uploadFile) {
        uploadFile.write(data, len);
    }
    if (final) {
        if (uploadFile) {
            uploadFile.close();
            uploadOk = true;
            Serial.printf("[Upload] Done: %u bytes\n", index + len);
            // Refresh RAM cache
            File f = LittleFS.open("/index.html", "r");
            if (f) {
                cachedIndexHtml = f.readString();
                f.close();
                Serial.printf("[FS] Recached: %u bytes\n", cachedIndexHtml.length());
            }
        }
    }
}

void handleDevColor(AsyncWebServerRequest *request) {
    if (request->hasParam("data", true)) {
        String hex = request->getParam("data", true)->value();
        uint16_t len = hex.length() / 2;
        if (len > NUM_PIXELS * 3) len = NUM_PIXELS * 3;
        for (uint16_t i = 0; i < len; i++) {
            char hi = hex.charAt(i * 2);
            char lo = hex.charAt(i * 2 + 1);
            uint8_t nib_hi = (hi >= 'a') ? (hi - 'a' + 10) : (hi >= 'A') ? (hi - 'A' + 10) : (hi - '0');
            uint8_t nib_lo = (lo >= 'a') ? (lo - 'a' + 10) : (lo >= 'A') ? (lo - 'A' + 10) : (lo - '0');
            devColorBuf[i] = (nib_hi << 4) | nib_lo;
        }
        state.effect = DEV_COLOR;
        devColorFresh = true;
    }
    request->send(200, "text/plain", "OK");
}

void handleStatus(AsyncWebServerRequest *request) {
    const char *effectName;
    switch (state.effect) {
        case RAINBOW:   effectName = "rainbow";   break;
        case GRADIENT:  effectName = "gradient";   break;
        case DEV_COLOR: effectName = "dev_color";  break;
        default:        effectName = "rainbow";    break;
    }
    String json = "{";
    json += "\"effect\":\"" + String(effectName) + "\",";
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
    request->send(200, "application/json", json);
}

void handleSetEffect(AsyncWebServerRequest *request) {
    if (request->hasParam("effect", true)) {
        String e = request->getParam("effect", true)->value();
        if (e == "rainbow") state.effect = RAINBOW;
        else if (e == "gradient") state.effect = GRADIENT;
        else if (e == "dev_color") state.effect = DEV_COLOR;
    }
    request->send(200, "text/plain", "OK");
}

void handleSetBrightness(AsyncWebServerRequest *request) {
    if (request->hasParam("brightness", true)) {
        int b = request->getParam("brightness", true)->value().toInt();
        if (b >= 0 && b <= 255) state.brightness = b;
    }
    request->send(200, "text/plain", "OK");
}

void handleSetCycleTime(AsyncWebServerRequest *request) {
    if (request->hasParam("cycletime", true)) {
        long ct = request->getParam("cycletime", true)->value().toInt();
        if (ct >= 1000 && ct <= 600000) state.cycleTimeMs = ct;
    }
    request->send(200, "text/plain", "OK");
}

void handleSetPalette(AsyncWebServerRequest *request) {
    if (request->hasParam("palette", true)) {
        String p = request->getParam("palette", true)->value();
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
    request->send(200, "text/plain", "OK");
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
        File f = LittleFS.open("/index.html", "r");
        if (f) {
            cachedIndexHtml = f.readString();
            f.close();
            Serial.printf("[FS] Cached index.html: %u bytes\n", cachedIndexHtml.length());
        }
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

    // Web server (async — no handleClient() needed in loop)
    DefaultHeaders::Instance().addHeader("Access-Control-Allow-Origin", "*");
    server.on("/", HTTP_GET, handleRoot);
    server.on("/source", HTTP_GET, handleSource);
    server.on("/upload", HTTP_GET, handleUploadPage);
    server.on("/upload", HTTP_POST, handleUploadComplete, handleUploadData);
    server.on("/status", HTTP_GET, handleStatus);
    server.on("/effect", HTTP_POST, handleSetEffect);
    server.on("/brightness", HTTP_POST, handleSetBrightness);
    server.on("/cycletime", HTTP_POST, handleSetCycleTime);
    server.on("/palette", HTTP_POST, handleSetPalette);
    server.on("/devcolor", HTTP_POST, handleDevColor);

    // OTA via ElegantOTA (async — no handle() needed in loop)
    ElegantOTA.begin(&server);
    ElegantOTA.onStart([]() { Serial.println("[OTA] Start"); });
    ElegantOTA.onEnd([](bool success) {
        Serial.printf("[OTA] %s\n", success ? "Done" : "Failed");
    });

    server.begin();
    Serial.println("[Web] Async server started on port 80");
    Serial.println("[OTA] ElegantOTA at /update");
}

// --- Main loop (pure render — no web/OTA blocking) ---

static uint32_t statsLastPrint = 0;
static uint32_t loopCount = 0;
static uint32_t loopTotalUs = 0;
static uint32_t loopMaxUs = 0;

void loop() {
    uint32_t loopStart = micros();

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

    // Phase accumulator for smooth animation
    static uint32_t lastFrameMs = 0;
    uint32_t now = millis();
    if (lastFrameMs == 0) lastFrameMs = now;
    uint32_t dt = now - lastFrameMs;
    lastFrameMs = now;
    if (state.cycleTimeMs > 0) {
        effectPhase += (float)dt / state.cycleTimeMs;
        if (effectPhase >= 1.0f) effectPhase -= (int)effectPhase;
    }

    for (uint8_t i = 0; i < NUM_STRIPS; i++) {
        // Brightness is applied via gammaHybrid inside render functions
        // (gamma on scalar brightness, not per-channel). Don't use
        // NeoPixel setBrightness() which would double-dim.
        strips[i].setBrightness(255);

        switch (state.effect) {
            case GRADIENT:
                renderGradient(strips[i], state);
                break;
            case DEV_COLOR:
                renderDevColor(strips[i], state);
                break;
            default: // RAINBOW
                renderRainbow(strips[i], state);
                break;
        }

        strips[i].show();
    }

    delay(16); // ~60fps

    // Loop timing stats
    uint32_t loopTime = micros() - loopStart;
    loopCount++;
    loopTotalUs += loopTime;
    if (loopTime > loopMaxUs) loopMaxUs = loopTime;
    if (now - statsLastPrint >= 2000) {
        float fps = loopCount * 1000.0f / (now - statsLastPrint);
        float avgMs = (loopTotalUs / 1000.0f) / loopCount;
        float maxMs = loopMaxUs / 1000.0f;
        Serial.printf("[Perf] FPS=%.1f  avg=%.1fms  max=%.1fms\n", fps, avgMs, maxMs);
        loopCount = 0;
        loopTotalUs = 0;
        loopMaxUs = 0;
        statsLastPrint = now;
    }
}
