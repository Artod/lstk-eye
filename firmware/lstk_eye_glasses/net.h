// WiFi + HTTP client for the laptop server.
//
// The server host is resolved once at connect time: a SERVER_HOST ending in
// ".local" goes through mDNS with SERVER_IP_FALLBACK as the backstop. All
// requests are plain blocking HTTP; /ask gets a longer timeout because it
// blocks through the full server pipeline.
#pragma once

#include <Arduino.h>

struct NetResult {
  bool ok = false;  // HTTP 2xx
  int code = 0;     // HTTP status, or a negative HTTPClient error
  String body;
};

// Connect to WiFi (blocking, WIFI_TIMEOUT_MS) and resolve the server host.
// status_cb receives short progress strings for the HUD. Used once in setup()
// so the user sees connect progress; afterwards loop() uses net_ensure().
bool net_connect(void (*status_cb)(const char*));

// Non-blocking connection keeper for loop(): returns true when WiFi is up
// (resolving the server host lazily if needed); when down, re-kicks
// WiFi.begin at most every 10 s and returns immediately. Never blocks, so
// the button FSM keeps running while offline.
bool net_ensure();

bool net_connected();

NetResult net_get(const String& path);
NetResult net_post_jpeg(const String& path, const uint8_t* buf, size_t len);
NetResult net_post_wav(const String& path, const uint8_t* buf, size_t len);
NetResult net_post_json(const String& path, const String& body);
