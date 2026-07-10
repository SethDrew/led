// Diagnostic only — reflash env:kaylab-knobs after use or the BS-26 stops broadcasting.
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <driver/adc.h>

Adafruit_ADS1115 ads;
bool adsOk = false;

void setup() {
    Serial.begin(921600);
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_4, ADC_ATTEN_DB_11);
    adc1_config_channel_atten(ADC1_CHANNEL_6, ADC_ATTEN_DB_11);
    adc1_config_channel_atten(ADC1_CHANNEL_7, ADC_ATTEN_DB_11);
    Wire.begin(23, 22);
    Wire.setClock(400000);
    adsOk = ads.begin(0x48, &Wire);
    if (adsOk) {
        ads.setGain(GAIN_ONE);
        ads.setDataRate(RATE_ADS1115_860SPS);
        ads.startADCReading(ADS1X15_REG_CONFIG_MUX_SINGLE_0, true);
    }
    Serial.printf("# raw_scan t_us,tens,ones,hundreds,ads0 ads=%d\n", adsOk);
}

void loop() {
    static int16_t adsVal = 0;
    static uint32_t lastAds = 0;
    uint32_t t = micros();
    int tens = adc1_get_raw(ADC1_CHANNEL_4);
    int ones = adc1_get_raw(ADC1_CHANNEL_6);
    int hund = adc1_get_raw(ADC1_CHANNEL_7);
    if (adsOk && t - lastAds >= 1163) {
        adsVal = ads.getLastConversionResults();
        lastAds = t;
    }
    Serial.printf("%lu,%d,%d,%d,%d\n", (unsigned long)t, tens, ones, hund, adsVal);
}
