// SSD1306 HUD: local status screens plus a renderer for server-sent scenes.
//
// Wire scenes (protocol v1) are flat JSON: {"v":1,"seq":N,"els":[...]} with
// elements {"t":"text",...} / {"t":"arrow",...} / {"t":"chevron",...}. All
// coordinates are pixels in the 128x64 panel space. A scene fully replaces
// the previous frame; the panel is redrawn only when seq changes.
#pragma once

#include <Arduino.h>

// Init the panel. mirrored=true flips the image horizontally to compensate
// the single reflection in the optical path.
bool hud_begin(bool mirrored);

// Local screens (not server-driven).
void hud_splash(const char* version);
void hud_message(const char* line1, const char* line2);
void hud_rec(uint32_t seconds, bool capped);
void hud_thinking();
void hud_error(const char* msg);
void hud_photo_count(int count);

// Parse a server response body and redraw when it carries a scene with a new
// seq. *active_out / *count_out are written only when the corresponding field
// is present in the response (pass nullptr to ignore). Returns false when the
// body is not valid JSON.
bool hud_apply_response(const String& body, bool* active_out, int* count_out);

// seq of the scene currently on the panel; sent back as last_seq so the
// server can answer "unchanged" with scene=null.
uint32_t hud_scene_seq();
