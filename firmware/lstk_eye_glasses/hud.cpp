#include "hud.h"

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>
#include <Wire.h>

#include "device_config.h"
#include "pins.h"

namespace {

constexpr int kWidth = 128;
constexpr int kHeight = 64;
constexpr int kArrowHeadPx = 6;

Adafruit_SSD1306 display(kWidth, kHeight, &Wire, -1);
uint32_t scene_seq = 0;

void frame_start() {
  // Every draw invalidates the last applied server scene: local screens
  // (errors, WiFi status, REC) clobber the panel, and the seq gate in
  // hud_apply_response must not suppress the redraw that restores it.
  // render_scene re-sets scene_seq after drawing.
  scene_seq = 0;
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextWrap(false);
  display.setTextSize(1);
}

// Arrow tip at (x, y), pointing along angle (degrees, screen convention:
// 0 = right, 90 = down). Shaft extends backward from the tip; the head is two
// strokes swept 45 degrees either side of the reversed shaft direction.
void draw_arrow(int x, int y, int angle_deg, int length) {
  const float rad = angle_deg * DEG_TO_RAD;
  const int tail_x = x - lroundf(cosf(rad) * length);
  const int tail_y = y - lroundf(sinf(rad) * length);
  display.drawLine(tail_x, tail_y, x, y, SSD1306_WHITE);
  for (int side = -1; side <= 1; side += 2) {
    const float head = (angle_deg + 180 + side * 45) * DEG_TO_RAD;
    const int hx = x + lroundf(cosf(head) * kArrowHeadPx);
    const int hy = y + lroundf(sinf(head) * kArrowHeadPx);
    display.drawLine(x, y, hx, hy, SSD1306_WHITE);
  }
}

// One V mark with its tip at (cx, cy) pointing along the axis-aligned
// direction (dx, dy).
void chevron_mark(int cx, int cy, int dx, int dy) {
  if (dx != 0) {
    display.drawLine(cx, cy, cx - 5 * dx, cy - 6, SSD1306_WHITE);
    display.drawLine(cx, cy, cx - 5 * dx, cy + 6, SSD1306_WHITE);
  } else {
    display.drawLine(cx, cy, cx - 6, cy - 5 * dy, SSD1306_WHITE);
    display.drawLine(cx, cy, cx + 6, cy - 5 * dy, SSD1306_WHITE);
  }
}

// Double chevron pointing along the given edge plus a short label beside it.
// tip_x/tip_y position the outer apex; -1 falls back to the physical panel
// edge (the server sets explicit coordinates when the optics crop the panel).
void draw_chevron(const char* edge, const char* label, int tip_x, int tip_y) {
  const int label_px = 6 * (int)strlen(label);
  int dx = 0, dy = 0;
  if (strcmp(edge, "right") == 0) {
    dx = 1;
    if (tip_x < 0) { tip_x = 126; tip_y = 32; }
  } else if (strcmp(edge, "left") == 0) {
    dx = -1;
    if (tip_x < 0) { tip_x = 1; tip_y = 32; }
  } else if (strcmp(edge, "up") == 0) {
    dy = -1;
    if (tip_x < 0) { tip_x = 64; tip_y = 1; }
  } else if (strcmp(edge, "down") == 0) {
    dy = 1;
    if (tip_x < 0) { tip_x = 64; tip_y = 62; }
  } else {
    return;
  }
  chevron_mark(tip_x, tip_y, dx, dy);
  chevron_mark(tip_x - 6 * dx, tip_y - 6 * dy, dx, dy);
  display.setTextSize(1);
  if (dx > 0) {
    display.setCursor(max(0, tip_x - 13 - label_px), tip_y - 4);
  } else if (dx < 0) {
    display.setCursor(tip_x + 14, tip_y - 4);
  } else if (dy < 0) {
    display.setCursor(max(0, tip_x - label_px / 2), tip_y + 13);
  } else {
    display.setCursor(max(0, tip_x - label_px / 2), tip_y - 20);
  }
  display.print(label);
}

// Object highlight: four corner brackets framing a square of half-size r
// around (x, y).
void draw_target(int x, int y, int r) {
  const int arm = max(3, r / 2);
  const int x0 = x - r, y0 = y - r, x1 = x + r, y1 = y + r;
  display.drawFastHLine(x0, y0, arm, SSD1306_WHITE);
  display.drawFastVLine(x0, y0, arm, SSD1306_WHITE);
  display.drawFastHLine(x1 - arm + 1, y0, arm, SSD1306_WHITE);
  display.drawFastVLine(x1, y0, arm, SSD1306_WHITE);
  display.drawFastHLine(x0, y1, arm, SSD1306_WHITE);
  display.drawFastVLine(x0, y1 - arm + 1, arm, SSD1306_WHITE);
  display.drawFastHLine(x1 - arm + 1, y1, arm, SSD1306_WHITE);
  display.drawFastVLine(x1, y1 - arm + 1, arm, SSD1306_WHITE);
}

void render_scene(JsonObject scene) {
  frame_start();
  JsonArray els = scene["els"].as<JsonArray>();
  for (JsonObject el : els) {
    const char* t = el["t"] | "";
    if (strcmp(t, "text") == 0) {
      display.setTextSize(el["size"] | 1);
      display.setCursor(el["x"] | 0, el["y"] | 0);
      display.print(el["text"] | "");
    } else if (strcmp(t, "arrow") == 0) {
      draw_arrow(el["x"] | 0, el["y"] | 0, el["angle"] | 225, el["length"] | 14);
    } else if (strcmp(t, "chevron") == 0) {
      draw_chevron(el["edge"] | "", el["label"] | "", el["x"] | -1, el["y"] | -1);
    } else if (strcmp(t, "target") == 0) {
      draw_target(el["x"] | 64, el["y"] | 32, el["r"] | 12);
    }
  }
  display.display();
}

}  // namespace

bool hud_begin(bool mirrored) {
  Wire.begin(PIN_OLED_SDA, PIN_OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDR)) {
    return false;
  }
  if (mirrored) {
    // The optical path has exactly one reflection, so the panel itself must
    // draw mirrored: SEGREMAP without remap (0xA0) undoes the horizontal
    // flip Adafruit's init applies; COMSCANDEC keeps the vertical direction.
    display.ssd1306_command(SSD1306_SEGREMAP);
    display.ssd1306_command(SSD1306_COMSCANDEC);
  }
  display.clearDisplay();
  display.display();
  return true;
}

void hud_splash(const char* version) {
  frame_start();
  display.setCursor(HUD_PAD_X, HUD_PAD_Y + 14);
  display.print("lstk-eye");
  display.setCursor(HUD_PAD_X, HUD_PAD_Y + 30);
  display.print("fw ");
  display.print(version);
  display.display();
}

void hud_message(const char* line1, const char* line2) {
  frame_start();
  display.setCursor(HUD_PAD_X, HUD_PAD_Y + 8);
  display.print(line1);
  display.setCursor(HUD_PAD_X, HUD_PAD_Y + 24);
  display.print(line2);
  display.display();
}

void hud_rec(uint32_t seconds, bool capped) {
  frame_start();
  display.fillCircle(HUD_PAD_X + 4, HUD_PAD_Y + 10, 4, SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(HUD_PAD_X + 14, HUD_PAD_Y + 2);
  display.print("REC");
  display.setCursor(HUD_PAD_X + 14, HUD_PAD_Y + 24);
  display.print(seconds);
  display.print("s");
  if (capped) {
    display.setTextSize(1);
    display.setCursor(HUD_PAD_X, 64 - HUD_PAD_Y - 8);
    display.print("max length");
  }
  display.display();
}

void hud_thinking() {
  frame_start();
  display.setCursor(HUD_PAD_X + 6, HUD_PAD_Y + 20);
  display.print("thinking...");
  display.display();
}

void hud_error(const char* msg) {
  frame_start();
  display.setCursor(HUD_PAD_X, HUD_PAD_Y + 8);
  display.print("! error");
  display.setTextWrap(true);
  display.setCursor(HUD_PAD_X, HUD_PAD_Y + 24);
  display.print(msg);
  // Local errors follow the same convention as server error scenes: the
  // exit is always stated.
  display.setTextWrap(false);
  display.setCursor(HUD_PAD_X, 64 - HUD_PAD_Y - 8);
  display.print("2click = reset");
  display.display();
}

void hud_photo_count(int count) {
  frame_start();
  display.setTextSize(2);
  display.setCursor(HUD_PAD_X + 2, HUD_PAD_Y + 14);
  display.print("x");
  display.print(count);
  display.setTextSize(1);
  display.setCursor(HUD_PAD_X + 2, HUD_PAD_Y + 34);
  display.print("offline count");
  display.display();
}

bool hud_apply_response(const String& body, bool* active_out, int* count_out) {
  JsonDocument doc;
  if (deserializeJson(doc, body) != DeserializationError::Ok) {
    return false;
  }
  if (active_out != nullptr && doc["active"].is<bool>()) {
    *active_out = doc["active"].as<bool>();
  }
  if (count_out != nullptr && doc["count"].is<int>()) {
    *count_out = doc["count"].as<int>();
  }
  JsonObject scene = doc["scene"];
  if (!scene.isNull()) {
    const uint32_t seq = scene["seq"] | 0;
    if (seq != scene_seq) {
      // render_scene's frame_start zeroes scene_seq; record the new seq after
      // drawing so the invalidation logic stays in one place.
      render_scene(scene);
      scene_seq = seq;
    }
  }
  return true;
}

uint32_t hud_scene_seq() {
  return scene_seq;
}
