// lstk-eye glasses firmware - thin HTTP client for the laptop server.
//
// Board: Seeed XIAO ESP32S3 Sense (FQBN esp32:esp32:XIAO_ESP32S3, PSRAM=opi).
// The device owns zero intelligence: camera in, pixels out. All thinking
// happens on the laptop; upgrading the server never touches this firmware.
//
// Gesture contract:
//   single click  - capture XGA photo, POST /photos, show returned scene
//   long press    - record mic while held; on release POST /ask with the WAV
//   double click  - POST /event {"type":"next"}
// While a session is active the loop POSTs a QVGA preview every
// PREVIEW_INTERVAL_MS and applies whatever scene comes back; a response with
// active=false ends the session and stops the previews.

#include <Arduino.h>

#include "buttons.h"
#include "cam.h"
#include "device_config.h"
#include "hud.h"
#include "mic.h"
#include "net.h"
#include "pins.h"

#define FW_VERSION "0.1.0"

// Top-level device state machine. WAITING is transient: handle_ask() blocks
// through the HTTP round trip and leaves it before returning to loop().
enum DeviceState : uint8_t { ST_IDLE, ST_RECORDING, ST_WAITING, ST_SESSION };

static DeviceState state = ST_IDLE;
static bool hardware_ok = false;
static int local_photo_count = 0;
static uint32_t last_preview_ms = 0;
static uint32_t last_rec_draw_ms = 0;

static String api(const char* path) {
  return String(path) + "?device_id=" + DEVICE_ID;
}

static String api_with_seq(const char* path) {
  return api(path) + "&last_seq=" + String(hud_scene_seq());
}

static void wifi_status(const char* msg) {
  hud_message("WiFi", msg);
}

static void handle_single_click() {
  camera_fb_t* fb = cam_capture_photo();
  if (fb == nullptr) {
    hud_error("capture failed");
    return;
  }
  NetResult r = net_post_jpeg(api("/api/v1/photos"), fb->buf, fb->len);
  cam_release(fb);
  if (r.ok) {
    int count = -1;
    hud_apply_response(r.body, nullptr, &count);
    if (count >= 0) {
      local_photo_count = count;
    }
  } else {
    // Server unreachable: keep counting locally so the HUD still gives
    // feedback; the buffer on the server is out of sync until the next ask.
    local_photo_count++;
    hud_photo_count(local_photo_count);
  }
}

static void handle_double_click() {
  NetResult r = net_post_json(api("/api/v1/event"), "{\"type\":\"next\"}");
  if (!r.ok) {
    hud_error("event failed");
    return;
  }
  bool active = (state == ST_SESSION);
  hud_apply_response(r.body, &active, nullptr);
  state = active ? ST_SESSION : ST_IDLE;
}

static void handle_ask() {
  uint8_t* wav = nullptr;
  const size_t len = mic_record_stop(&wav);
  if (len == 0 || wav == nullptr) {
    hud_error("no audio captured");
    state = ST_IDLE;
    return;
  }
  state = ST_WAITING;
  hud_thinking();
  NetResult r = net_post_wav(api("/api/v1/ask"), wav, len);
  if (!r.ok) {
    hud_error("ask failed");
    state = ST_IDLE;
    return;
  }
  bool active = true;
  hud_apply_response(r.body, &active, nullptr);
  local_photo_count = 0;  // the server consumed the buffered photos
  state = active ? ST_SESSION : ST_IDLE;
  last_preview_ms = millis();
}

static void send_preview() {
  camera_fb_t* fb = cam_capture_preview();
  if (fb == nullptr) {
    return;
  }
  NetResult r = net_post_jpeg(api_with_seq("/api/v1/preview"), fb->buf, fb->len);
  cam_release(fb);
  if (!r.ok) {
    return;  // transient failure; next preview retries
  }
  bool active = true;
  hud_apply_response(r.body, &active, nullptr);
  if (!active) {
    state = ST_IDLE;
  }
}

void setup() {
  Serial.begin(115200);
  if (!hud_begin(HUD_MIRRORED != 0)) {
    Serial.println("[hud] SSD1306 init failed");
  }
  hud_splash(FW_VERSION);
  buttons_begin(PIN_BUTTON);
  if (!cam_begin()) {
    hud_error("camera init failed");
    return;
  }
  if (!mic_begin()) {
    hud_error("mic init failed (PSRAM?)");
    return;
  }
  hardware_ok = true;
  if (!net_connect(wifi_status)) {
    return;  // loop() keeps retrying
  }
  NetResult health = net_get("/api/v1/health");
  hud_message("lstk-eye " FW_VERSION,
              health.ok ? "server ok" : "server unreachable");
}

void loop() {
  if (!hardware_ok) {
    delay(1000);
    return;
  }
  // Recording is deliberately not interrupted by a WiFi drop; the ask will
  // fail visibly instead.
  if (state != ST_RECORDING && !net_connected()) {
    hud_message("WiFi", "reconnecting...");
    if (!net_connect(wifi_status)) {
      delay(2000);
      return;
    }
  }

  const ButtonEvent ev = buttons_poll();
  switch (state) {
    case ST_IDLE:
    case ST_SESSION:
      if (ev == BTN_SINGLE) {
        handle_single_click();
      } else if (ev == BTN_DOUBLE) {
        handle_double_click();
      } else if (ev == BTN_LONG_START) {
        if (mic_record_start()) {
          state = ST_RECORDING;
          last_rec_draw_ms = millis();
          hud_rec(0, false);
        } else {
          hud_error("mic start failed");
        }
      }
      // Previews are optional traffic: never start one while the button is
      // held or a gesture is mid-flight, because the blocking POST would
      // swallow the next edges (a double click would decay into a single
      // click). This narrows, but does not close, the window where a click
      // lands during an in-flight request; the full fix is moving HTTP to a
      // worker task.
      if (state == ST_SESSION && !buttons_busy() &&
          millis() - last_preview_ms >= PREVIEW_INTERVAL_MS) {
        last_preview_ms = millis();
        send_preview();
      }
      break;

    case ST_RECORDING:
      mic_record_poll();
      if (millis() - last_rec_draw_ms >= 500) {
        last_rec_draw_ms = millis();
        hud_rec(mic_record_seconds(), mic_record_full());
      }
      if (ev == BTN_LONG_RELEASE) {
        handle_ask();
      }
      break;

    case ST_WAITING:
      // Unreachable steady state; see handle_ask().
      state = ST_IDLE;
      break;
  }
}
