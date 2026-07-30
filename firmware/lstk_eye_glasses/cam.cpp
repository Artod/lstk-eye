#include "cam.h"

#include <Arduino.h>

#include "ESP32_OV5640_AF.h"
#include "pins.h"

namespace {

OV5640 ov5640;
framesize_t current_size = FRAMESIZE_XGA;

int width_for(framesize_t size) {
  switch (size) {
    case FRAMESIZE_QVGA:
      return 320;
    case FRAMESIZE_XGA:
      return 1024;
    default:
      return 0;
  }
}

camera_fb_t* capture_at(framesize_t size) {
  sensor_t* s = esp_camera_sensor_get();
  if (s == nullptr) {
    return nullptr;
  }
  if (size != current_size) {
    if (s->set_framesize(s, size) != 0) {
      return nullptr;
    }
    current_size = size;
  }
  // With fb_count=2 + CAMERA_GRAB_LATEST the driver free-runs, so right after
  // a size switch the queue can still hold frames at the OLD resolution (or
  // one torn during sensor reconfig). Returning one of those would feed the
  // planner a QVGA preview instead of the XGA photo. Flush until the frame
  // actually matches the requested size.
  const int want_w = width_for(size);
  for (int attempt = 0; attempt < 4; ++attempt) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb == nullptr) {
      return nullptr;
    }
    if (want_w == 0 || (int)fb->width == want_w) {
      return fb;
    }
    esp_camera_fb_return(fb);
  }
  return nullptr;
}

}  // namespace

bool cam_begin() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_pwdn = CAM_PIN_PWDN;
  config.pin_reset = CAM_PIN_RESET;
  config.pin_xclk = CAM_PIN_XCLK;
  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_d7 = CAM_PIN_Y9;
  config.pin_d6 = CAM_PIN_Y8;
  config.pin_d5 = CAM_PIN_Y7;
  config.pin_d4 = CAM_PIN_Y6;
  config.pin_d3 = CAM_PIN_Y5;
  config.pin_d2 = CAM_PIN_Y4;
  config.pin_d1 = CAM_PIN_Y3;
  config.pin_d0 = CAM_PIN_Y2;
  config.pin_vsync = CAM_PIN_VSYNC;
  config.pin_href = CAM_PIN_HREF;
  config.pin_pclk = CAM_PIN_PCLK;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_XGA;
  config.jpeg_quality = 12;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&config) != ESP_OK) {
    return false;
  }
  current_size = FRAMESIZE_XGA;

  // The OV5640 autofocus MCU is empty at power-on; upload its firmware and
  // switch to continuous autofocus. Failure is non-fatal (fixed focus).
  sensor_t* s = esp_camera_sensor_get();
  if (ov5640.start(s)) {
    if (ov5640.focusInit() == 0 && ov5640.autoFocusMode() == 0) {
      Serial.println("[cam] OV5640 continuous autofocus enabled");
    } else {
      Serial.println("[cam] OV5640 AF init failed, running fixed focus");
    }
  } else {
    Serial.println("[cam] sensor is not an OV5640, skipping AF");
  }
  return true;
}

camera_fb_t* cam_capture_photo() {
  return capture_at(FRAMESIZE_XGA);
}

camera_fb_t* cam_capture_preview() {
  return capture_at(FRAMESIZE_QVGA);
}

void cam_release(camera_fb_t* fb) {
  if (fb != nullptr) {
    esp_camera_fb_return(fb);
  }
}
