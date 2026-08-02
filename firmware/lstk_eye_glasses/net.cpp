#include "net.h"

#include <ESPmDNS.h>
#include <HTTPClient.h>
#include <WiFi.h>

#include "device_config.h"

namespace {

String base_url;
uint32_t last_begin_ms = 0;

String resolve_host() {
  String host = SERVER_HOST;
  if (!host.endsWith(".local")) {
    return host;
  }
  const String name = host.substring(0, host.length() - 6);
  if (MDNS.begin("lstk-glasses")) {
    const IPAddress ip = MDNS.queryHost(name);
    if (ip != IPAddress()) {
      return ip.toString();
    }
  }
  return SERVER_IP_FALLBACK;
}

uint32_t timeout_for(const String& path) {
  return path.startsWith("/api/v1/ask") ? HTTP_TIMEOUT_ASK_MS : HTTP_TIMEOUT_MS;
}

NetResult request(const String& path, const char* content_type,
                  const uint8_t* payload, size_t len) {
  NetResult r;
  if (WiFi.status() != WL_CONNECTED || base_url.length() == 0) {
    r.code = HTTPC_ERROR_CONNECTION_REFUSED;
    return r;
  }
  HTTPClient http;
  http.setConnectTimeout(3000);
  http.setTimeout(timeout_for(path));
  if (!http.begin(base_url + path)) {
    r.code = HTTPC_ERROR_CONNECTION_REFUSED;
    return r;
  }
  int code;
  if (content_type == nullptr) {
    code = http.GET();
  } else {
    http.addHeader("Content-Type", content_type);
    // HTTPClient::POST takes a non-const pointer but does not modify the body.
    code = http.POST(const_cast<uint8_t*>(payload), len);
  }
  r.code = code;
  if (code > 0) {
    r.body = http.getString();
  }
  r.ok = code >= 200 && code < 300;
  http.end();
  return r;
}

}  // namespace

bool net_connect(void (*status_cb)(const char*)) {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  if (status_cb != nullptr) {
    status_cb("connecting...");
  }
  const uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start > WIFI_TIMEOUT_MS) {
      if (status_cb != nullptr) {
        status_cb("connect timeout");
      }
      return false;
    }
    delay(100);
  }
  const String host = resolve_host();
  base_url = String("http://") + host + ":" + SERVER_PORT;
  if (status_cb != nullptr) {
    status_cb(("server " + host).c_str());
  }
  return true;
}

bool net_connected() {
  return WiFi.status() == WL_CONNECTED;
}

bool net_ensure() {
  if (WiFi.status() == WL_CONNECTED) {
    if (base_url.length() == 0) {
      // First moment of connectivity (e.g. the blocking setup() attempt timed
      // out and the link came up later): resolve the server host now.
      const String host = resolve_host();
      base_url = String("http://") + host + ":" + SERVER_PORT;
    }
    return true;
  }
  const uint32_t now = millis();
  // Re-kick the connection at most every 10 s. Calling WiFi.begin while a
  // previous attempt is still in flight logs "cannot set config" noise, so
  // disconnect first and space the attempts out.
  if (last_begin_ms == 0 || now - last_begin_ms >= 10000) {
    last_begin_ms = now;
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASS);
  }
  return false;
}

NetResult net_get(const String& path) {
  return request(path, nullptr, nullptr, 0);
}

NetResult net_post_jpeg(const String& path, const uint8_t* buf, size_t len) {
  return request(path, "image/jpeg", buf, len);
}

NetResult net_post_wav(const String& path, const uint8_t* buf, size_t len) {
  return request(path, "audio/wav", buf, len);
}

NetResult net_post_json(const String& path, const String& body) {
  return request(path, "application/json", (const uint8_t*)body.c_str(),
                 body.length());
}
