// Pin map for the lstk-eye glasses build on a Seeed XIAO ESP32S3 Sense.
#pragma once

// Camera socket on the Sense expansion board (OV5640 swapped in for the
// stock OV2640). Standard CAMERA_MODEL_XIAO_ESP32S3 pin set from the
// CameraWebServer example.
#define CAM_PIN_PWDN -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK 10
#define CAM_PIN_SIOD 40
#define CAM_PIN_SIOC 39
#define CAM_PIN_Y9 48
#define CAM_PIN_Y8 11
#define CAM_PIN_Y7 12
#define CAM_PIN_Y6 14
#define CAM_PIN_Y5 16
#define CAM_PIN_Y4 18
#define CAM_PIN_Y3 17
#define CAM_PIN_Y2 15
#define CAM_PIN_VSYNC 38
#define CAM_PIN_HREF 47
#define CAM_PIN_PCLK 13

// SSD1306 128x64 OLED on I2C. D4/D5 are the XIAO variant aliases for
// GPIO5/GPIO6 (also the default Wire pins on this board).
#define PIN_OLED_SDA D4
#define PIN_OLED_SCL D5
#define OLED_I2C_ADDR 0x3C

// Panel-mount button between D1 (GPIO2) and GND; INPUT_PULLUP, active low.
#define PIN_BUTTON D1

// Onboard PDM microphone.
#define PIN_PDM_CLK 42
#define PIN_PDM_DATA 41
