// Debounced non-blocking button gesture FSM.
//
// Gestures: single click, double click, long-press start, long-press release.
// Timings come from device_config.h (30 ms debounce, 600 ms long-press
// threshold, 350 ms double-click gap - bench-validated).
#pragma once

#include <Arduino.h>

enum ButtonEvent : uint8_t {
  BTN_NONE = 0,
  BTN_SINGLE,
  BTN_DOUBLE,
  BTN_LONG_START,
  BTN_LONG_RELEASE,
};

void buttons_begin(uint8_t pin);

// Call every loop() iteration. Returns at most one event per call.
ButtonEvent buttons_poll();

// True while the button is held or a gesture is mid-flight (e.g. inside the
// double-click gap). Blocking work started in that window would swallow
// gesture edges, so callers defer optional network calls while busy.
bool buttons_busy();
