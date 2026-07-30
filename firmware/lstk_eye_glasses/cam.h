// OV5640 camera: JPEG capture at XGA (photos) and QVGA (session previews).
#pragma once

#include "esp_camera.h"

// Init the camera (JPEG, frame buffers in PSRAM) and upload the OV5640
// autofocus firmware - the sensor boots with autofocus dead until then.
bool cam_begin();

// Capture one JPEG frame, switching the sensor resolution when needed.
// Returns nullptr on failure; release with cam_release().
camera_fb_t* cam_capture_photo();    // XGA 1024x768
camera_fb_t* cam_capture_preview();  // QVGA 320x240

void cam_release(camera_fb_t* fb);
