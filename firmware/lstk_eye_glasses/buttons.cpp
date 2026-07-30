#include "buttons.h"

#include "device_config.h"

namespace {

uint8_t btn_pin = 0;

// Debounce layer: a raw level change must hold for BTN_DEBOUNCE_MS before it
// becomes the stable level the gesture layer sees.
bool raw_last = false;      // true = pressed (pin low)
bool stable_pressed = false;
uint32_t raw_change_ms = 0;

// Gesture layer over the debounced level.
enum GestureState : uint8_t {
  G_IDLE,    // released, nothing pending
  G_PRESS1,  // first press held, waiting for release or long-press threshold
  G_LONG,    // long press active, waiting for release
  G_GAP,     // released after a short press, waiting for a second press
  G_PRESS2,  // second press held, release completes the double click
};
GestureState gstate = G_IDLE;
uint32_t t_edge = 0;  // millis() of the edge that entered the current state

}  // namespace

void buttons_begin(uint8_t pin) {
  btn_pin = pin;
  pinMode(pin, INPUT_PULLUP);
  raw_last = (digitalRead(pin) == LOW);
  stable_pressed = raw_last;
  raw_change_ms = millis();
}

ButtonEvent buttons_poll() {
  const uint32_t now = millis();

  const bool raw = (digitalRead(btn_pin) == LOW);
  if (raw != raw_last) {
    raw_last = raw;
    raw_change_ms = now;
  }
  bool edge = false;
  if (raw != stable_pressed && (now - raw_change_ms) >= BTN_DEBOUNCE_MS) {
    stable_pressed = raw;
    edge = true;
  }

  switch (gstate) {
    case G_IDLE:
      if (edge && stable_pressed) {
        gstate = G_PRESS1;
        t_edge = now;
      }
      break;
    case G_PRESS1:
      if (edge && !stable_pressed) {
        gstate = G_GAP;
        t_edge = now;
      } else if (now - t_edge >= BTN_LONGPRESS_MS) {
        gstate = G_LONG;
        return BTN_LONG_START;
      }
      break;
    case G_LONG:
      if (edge && !stable_pressed) {
        gstate = G_IDLE;
        return BTN_LONG_RELEASE;
      }
      break;
    case G_GAP:
      if (edge && stable_pressed) {
        gstate = G_PRESS2;
        t_edge = now;
      } else if (now - t_edge >= BTN_DOUBLECLICK_MS) {
        gstate = G_IDLE;
        return BTN_SINGLE;
      }
      break;
    case G_PRESS2:
      if (edge && !stable_pressed) {
        gstate = G_IDLE;
        return BTN_DOUBLE;
      }
      break;
  }
  return BTN_NONE;
}
