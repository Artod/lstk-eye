// Device configuration template.
//
// Copy this file to device_config.h in the same directory and fill in your
// network. device_config.h is gitignored - credentials never enter the repo.
#pragma once

// 2.4 GHz network only; the ESP32-S3 does not see 5 GHz.
#define WIFI_SSID "your-2.4ghz-ssid"
#define WIFI_PASS "your-password"

// Laptop server. A host ending in ".local" is resolved via mDNS at boot;
// SERVER_IP_FALLBACK is used when the query fails.
#define SERVER_HOST "lstk-eye.local"
#define SERVER_IP_FALLBACK "192.168.1.50"
#define SERVER_PORT 8321
#define DEVICE_ID "glasses"

// 1 = initialize the panel horizontally mirrored to compensate the single
// beamsplitter reflection; 0 for bench tests without the optics.
#define HUD_MIRRORED 1

// Button gesture timing (bench-validated).
#define BTN_DEBOUNCE_MS 30
#define BTN_LONGPRESS_MS 600
#define BTN_DOUBLECLICK_MS 350

// Session preview cadence (~2.5 fps) and recording cap.
#define PREVIEW_INTERVAL_MS 400
#define REC_MAX_SECONDS 20

// Network timeouts. /ask blocks through the full server pipeline, so it gets
// its own, longer budget.
#define WIFI_TIMEOUT_MS 20000
#define HTTP_TIMEOUT_MS 5000
// Must stay above the server planner timeout (25 s default) so the device
// receives the error scene instead of timing out first.
#define HTTP_TIMEOUT_ASK_MS 30000
