// Optics alignment aid: light every pixel of the SSD1306 at full brightness.
//
// Use this while positioning the lens/beamsplitter to see the exact extent of
// the panel's active area (21.7 x 10.9 mm emitting surface) through the
// optical path. Flash the main firmware back when done.
//
// Board: XIAO ESP32S3 (any PSRAM setting). OLED on D4/D5, addr 0x3C.

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>

Adafruit_SSD1306 display(128, 64, &Wire, -1);

void setup() {
  Wire.begin(D4, D5);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  // Max contrast: full charge-pump drive for the brightest possible field.
  display.ssd1306_command(SSD1306_SETCONTRAST);
  display.ssd1306_command(0xFF);
  display.fillRect(0, 0, 128, 64, SSD1306_WHITE);
  display.display();
}

void loop() {}
