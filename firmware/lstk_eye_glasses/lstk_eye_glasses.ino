// lstk-eye glasses firmware - thin HTTP client for the laptop server.
//
// Board: Seeed XIAO ESP32S3 Sense (FQBN esp32:esp32:XIAO_ESP32S3, PSRAM=opi).
// The device owns zero intelligence: camera in, pixels out. All thinking
// happens on the laptop; upgrading the server never touches this firmware.
//
// Gesture contract:
//   single click  - capture XGA photo, POST /photos, show returned scene
//   long press    - record mic while held; on release POST /ask with the WAV
//   double click  - POST /event {"type":"reset"} - end the chat, clear all
//
// Concurrency: all networking and camera captures run on a dedicated worker
// task (core 0) fed by a job queue, so the UI loop polls the button every
// millisecond no matter what is in flight - a click can never be swallowed
// by a blocking HTTP request. Results come back through a queue and are
// applied (all HUD drawing) on the UI loop only. While a session is active
// the worker self-schedules a QVGA preview every PREVIEW_INTERVAL_MS.

#include <Arduino.h>
#include <esp_system.h>

#include "buttons.h"
#include "cam.h"
#include "device_config.h"
#include "hud.h"
#include "mic.h"
#include "net.h"
#include "pins.h"

#define FW_VERSION "0.2.0"

// UI-side device state. WAITING = an ask is in flight on the worker.
enum DeviceState : uint8_t { ST_IDLE, ST_RECORDING, ST_WAITING, ST_SESSION };

enum JobType : uint8_t { JOB_PHOTO, JOB_RESET, JOB_ASK };

struct NetJob {
  JobType type;
  uint8_t* wav;  // JOB_ASK payload, freed by the worker after the POST
  size_t len;
};

struct NetResultMsg {
  JobType type;
  bool ok;
  char body[2048];
};

static DeviceState state = ST_IDLE;
static bool hardware_ok = false;
static int local_photo_count = 0;
static uint32_t last_rec_draw_ms = 0;

static QueueHandle_t q_jobs;
static QueueHandle_t q_results;
// Written by the UI loop, read by the worker to self-schedule previews; the
// worker also clears it when a preview response says the session ended.
static volatile bool g_session_active = false;
static volatile bool g_online = false;
// True from long-press start until the ask result is applied: the worker
// sends no previews and the UI ignores stale preview scenes, so REC and
// "thinking..." screens are never overdrawn by tracking frames.
static volatile bool g_hold_previews = false;

static String api(const char* path) {
  return String(path) + "?device_id=" + DEVICE_ID;
}

static String api_with_seq(const char* path) {
  return api(path) + "&last_seq=" + String(hud_scene_seq());
}

static void wifi_status(const char* msg) {
  hud_message("WiFi", msg);
}

// --- worker task (core 0): all camera captures and HTTP ---

static void push_result(JobType type, bool ok, const String& body) {
  // Static: a 2 KB struct has no business on the worker's stack.
  static NetResultMsg msg;
  msg.type = type;
  msg.ok = ok;
  strlcpy(msg.body, body.c_str(), sizeof(msg.body));
  xQueueSend(q_results, &msg, 0);  // drop when the UI is behind
}

static void run_job(const NetJob& job) {
  switch (job.type) {
    case JOB_PHOTO: {
      camera_fb_t* fb = cam_capture_photo();
      if (fb == nullptr) {
        push_result(JOB_PHOTO, false, "");
        return;
      }
      NetResult r = net_post_jpeg(api("/api/v1/photos"), fb->buf, fb->len);
      cam_release(fb);
      push_result(JOB_PHOTO, r.ok, r.body);
      return;
    }
    case JOB_RESET: {
      NetResult r = net_post_json(api("/api/v1/event"), "{\"type\":\"reset\"}");
      push_result(JOB_RESET, r.ok, r.body);
      return;
    }
    case JOB_ASK: {
      NetResult r = net_post_wav(api("/api/v1/ask"), job.wav, job.len);
      free(job.wav);
      push_result(JOB_ASK, r.ok, r.body);
      return;
    }
  }
}

static void worker_task(void*) {
  uint32_t last_preview_ms = 0;
  for (;;) {
    g_online = net_ensure();
    NetJob job;
    if (xQueueReceive(q_jobs, &job, pdMS_TO_TICKS(20)) == pdTRUE) {
      if (g_online) {
        run_job(job);
      } else {
        if (job.type == JOB_ASK) {
          free(job.wav);
        }
        push_result(job.type, false, "");
      }
      continue;
    }
    if (g_online && g_session_active && !g_hold_previews &&
        millis() - last_preview_ms >= PREVIEW_INTERVAL_MS &&
        uxQueueMessagesWaiting(q_jobs) == 0) {
      last_preview_ms = millis();
      camera_fb_t* fb = cam_capture_preview();
      if (fb == nullptr) {
        continue;
      }
      NetResult r = net_post_jpeg(api_with_seq("/api/v1/preview"), fb->buf, fb->len);
      cam_release(fb);
      if (r.ok && r.body.length() > 0) {
        static NetResultMsg msg;
        msg.type = (JobType)0xFF;  // preview marker, see apply_result
        msg.ok = true;
        strlcpy(msg.body, r.body.c_str(), sizeof(msg.body));
        xQueueSend(q_results, &msg, 0);
      }
    }
  }
}

// --- UI side: state transitions and drawing ---

static void enqueue(JobType type, uint8_t* wav = nullptr, size_t len = 0) {
  NetJob job{type, wav, len};
  if (xQueueSend(q_jobs, &job, 0) != pdTRUE && type == JOB_ASK) {
    free(wav);  // queue jammed; never leak the recording
    hud_error("busy, retry");
    state = ST_IDLE;
  }
}

static void apply_result(const NetResultMsg& msg) {
  const String body(msg.body);
  if (msg.type == (JobType)0xFF) {  // preview
    if (state == ST_RECORDING || state == ST_WAITING) {
      return;  // stale frame from before the hold engaged
    }
    bool active = true;
    hud_apply_response(body, &active, nullptr);
    if (!active) {
      g_session_active = false;
      if (state == ST_SESSION) {
        state = ST_IDLE;
      }
    }
    return;
  }
  switch (msg.type) {
    case JOB_PHOTO:
      if (msg.ok) {
        int count = -1;
        hud_apply_response(body, nullptr, &count);
        if (count >= 0) {
          local_photo_count = count;
        }
      } else {
        // Server unreachable: count locally so the HUD still gives feedback.
        local_photo_count++;
        hud_photo_count(local_photo_count);
      }
      break;
    case JOB_RESET: {
      g_hold_previews = false;
      if (!msg.ok) {
        hud_error("reset failed");
        break;
      }
      bool active = false;
      hud_apply_response(body, &active, nullptr);
      local_photo_count = 0;
      g_session_active = active;
      state = active ? ST_SESSION : ST_IDLE;
      break;
    }
    case JOB_ASK: {
      g_hold_previews = false;
      if (!msg.ok) {
        hud_error("ask failed");
        state = ST_IDLE;
        g_session_active = false;
        break;
      }
      bool active = true;
      hud_apply_response(body, &active, nullptr);
      local_photo_count = 0;
      g_session_active = active;
      state = active ? ST_SESSION : ST_IDLE;
      break;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(50);
  // 1=power-on, 3=sw, 4=panic, 5=int_wdt, 6=task_wdt, 7=other_wdt, 9=brownout
  Serial.printf("[boot] fw %s, reset reason %d\n", FW_VERSION, (int)esp_reset_reason());
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

  q_jobs = xQueueCreate(4, sizeof(NetJob));
  q_results = xQueueCreate(4, sizeof(NetResultMsg));

  // Initial blocking connect keeps boot feedback simple; afterwards the
  // worker owns WiFi and all requests.
  if (net_connect(wifi_status)) {
    NetResult health = net_get("/api/v1/health");
    hud_message("lstk-eye " FW_VERSION,
                health.ok ? "server ok" : "server unreachable");
  }
  xTaskCreatePinnedToCore(worker_task, "net", 20480, nullptr, 1, nullptr, 0);
}

void loop() {
  if (!hardware_ok) {
    delay(1000);
    return;
  }
  static bool was_online = true;
  const bool online = g_online;
  if (online != was_online) {
    was_online = online;
    if (!online && state != ST_RECORDING) {
      hud_message("WiFi down", "retrying in bg");
    } else if (online && state == ST_IDLE) {
      hud_message("WiFi", "connected");
    }
  }

  const ButtonEvent ev = buttons_poll();
  switch (state) {
    case ST_IDLE:
    case ST_SESSION:
      if (ev == BTN_SINGLE) {
        enqueue(JOB_PHOTO);
      } else if (ev == BTN_DOUBLE) {
        enqueue(JOB_RESET);
      } else if (ev == BTN_LONG_START) {
        if (mic_record_start()) {
          g_hold_previews = true;
          state = ST_RECORDING;
          last_rec_draw_ms = millis();
          hud_rec(0, false);
        } else {
          hud_error("mic start failed");
        }
      }
      break;

    case ST_RECORDING:
      mic_record_poll();
      if (millis() - last_rec_draw_ms >= 500) {
        last_rec_draw_ms = millis();
        hud_rec(mic_record_seconds(), mic_record_full());
      }
      if (ev == BTN_LONG_RELEASE) {
        uint8_t* wav = nullptr;
        const size_t len = mic_record_stop(&wav);
        if (len == 0 || wav == nullptr) {
          hud_error("no audio captured");
          state = ST_IDLE;
          g_hold_previews = false;
        } else {
          state = ST_WAITING;
          hud_thinking();
          enqueue(JOB_ASK, wav, len);
        }
      }
      break;

    case ST_WAITING:
      // The ask runs on the worker; a double click can still bail out.
      if (ev == BTN_DOUBLE) {
        enqueue(JOB_RESET);
      }
      break;
  }

  static NetResultMsg msg;
  while (xQueueReceive(q_results, &msg, 0) == pdTRUE) {
    apply_result(msg);
  }
  delay(1);
}
