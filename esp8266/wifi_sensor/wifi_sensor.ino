#include <Wire.h>
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Adafruit_BMP085.h>

#include "config.h"

// ===== ОБЪЕКТЫ =====
Adafruit_BMP085 bmp;
ESP8266WebServer server(80);

// ===== ДАННЫЕ =====
float temperature;
int32_t pressure;
float altitude;
int32_t seaLevel;
float realAltitude;
int analogValue;

// ===== LED =====
unsigned long lastBlink = 0;
const unsigned long BLINK_INTERVAL = 5000; // 5 сек

// ===== WIFI =====
void setupWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(STA_SSID, STA_PASS);

  Serial.print("Connecting to WiFi");

  unsigned long startAttempt = millis();

  while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < 10000) {

    // --- МИГАНИЕ ПРИ ПОДКЛЮЧЕНИИ ---
    digitalWrite(LED_BUILTIN, LOW);   // включить
    delay(150);
    digitalWrite(LED_BUILTIN, HIGH);  // выключить
    delay(150);

    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi failed, starting AP");
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASS);
    Serial.print("AP IP: ");
    Serial.println(WiFi.softAPIP());
  }
}

// ===== JSON =====
String makeJSON() {
  String json = "{";
  json += "\"temperature\":" + String(temperature) + ",";
  json += "\"pressure\":" + String(pressure) + ",";
  json += "\"altitude\":" + String(altitude) + ",";
  json += "\"sea_level\":" + String(seaLevel) + ",";
  json += "\"real_altitude\":" + String(realAltitude) + ",";
  json += "\"a0\":" + String(analogValue);
  json += "}";
  return json;
}

// ===== HTTP =====
void handleRoot() {
  server.send(200, "application/json", makeJSON());
}

void setup() {
  Serial.begin(9600);
  delay(500);

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);   // выключить (активный LOW)

  setupWiFi();

  if (!bmp.begin()) {
    Serial.println("BMP085 not found!");
    while (1);
  }

  server.on("/", handleRoot);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  // ===== СЧИТЫВАНИЕ =====
  temperature = bmp.readTemperature();
  pressure = bmp.readPressure();
  altitude = bmp.readAltitude();
  seaLevel = bmp.readSealevelPressure();
  realAltitude = bmp.readAltitude(101500);
  analogValue = analogRead(A0);

  // ===== SERIAL CSV =====
  Serial.print("!");
  Serial.print(temperature); Serial.print(";");
  Serial.print(pressure); Serial.print(";");
  Serial.print(altitude); Serial.print(";");
  Serial.print(seaLevel); Serial.print(";");
  Serial.print(realAltitude); Serial.print(";");
  Serial.print(analogValue);
  Serial.println();

  server.handleClient();

  // ===== МИГАНИЕ КАЖДЫЕ 5 СЕК =====
  if (millis() - lastBlink >= BLINK_INTERVAL) {
    lastBlink = millis();
    digitalWrite(LED_BUILTIN, LOW);   // включить
    delay(120);
    digitalWrite(LED_BUILTIN, HIGH);  // выключить
  }

  delay(50);
}
