# lstk-eye glasses firmware

Firmware for the glasses unit: a Seeed XIAO ESP32S3 Sense acting as a thin
HTTP client to the laptop server. Camera in, pixels out - all intelligence
lives on the laptop.

Sketch: `lstk_eye_glasses/`. Every `.h`/`.cpp` in that directory is compiled
as one unit by the Arduino build.

| File | Role |
| --- | --- |
| `lstk_eye_glasses.ino` | setup, main loop, device state machine (IDLE / RECORDING / WAITING / SESSION) |
| `device_config.example.h` | template for `device_config.h` (WiFi credentials, server host, flags, timings) |
| `pins.h` | camera / OLED / button / mic pin map |
| `hud.h/.cpp` | SSD1306 init (incl. optical mirror flag), wire-scene renderer, local status screens |
| `buttons.h/.cpp` | debounced gesture FSM: single / double / long-press start / long-press release |
| `cam.h/.cpp` | OV5640 init + autofocus firmware upload, XGA photo and QVGA preview capture |
| `mic.h/.cpp` | PDM hold-to-talk recording into PSRAM, WAV wrapping |
| `net.h/.cpp` | WiFi connect, mDNS resolve, HTTP POST/GET helpers |

## Configure

```sh
cp lstk_eye_glasses/device_config.example.h lstk_eye_glasses/device_config.h
```

Edit `device_config.h`: WiFi SSID/password (2.4 GHz network), server host
(`lstk-eye.local` resolves via mDNS, with `SERVER_IP_FALLBACK` as backstop),
and `HUD_MIRRORED` (1 when the OLED is viewed through the beamsplitter, 0 for
bench tests without optics). `device_config.h` is gitignored - credentials
never enter the repo.

## Arduino IDE 2 setup

1. **File > Preferences > Additional boards manager URLs**, add:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
2. **Tools > Board > Boards Manager**, install **"esp32 by Espressif
   Systems"** (tested with 3.3.11). NOT the "Arduino ESP32 Boards" package -
   the XIAO variant is not in it.
3. **Tools > Board > esp32 > XIAO_ESP32S3**.
4. **Tools > PSRAM > "OPI PSRAM"** - mandatory; the camera frame buffers and
   the audio buffer live in PSRAM.
5. **Tools > Port** - looks like `/dev/cu.usbmodemXXXX` on macOS (never
   `cu.debug-console`).
6. Serial Monitor at **115200** baud.

## Libraries

Install via **Tools > Manage Libraries**:

| Library | Tested version |
| --- | --- |
| Adafruit SSD1306 | 2.5.17 |
| Adafruit GFX Library | 1.12.6 |
| Adafruit BusIO | 1.17.4 |
| ArduinoJson | 7.4.3 |
| OV5640 Auto Focus for ESP32 Camera | 0.1.1 |

The camera driver (`esp_camera.h`) and I2S (`ESP_I2S.h`) ship with the esp32
core; no extra install.

## arduino-cli equivalents

```sh
arduino-cli config add board_manager.additional_urls \
  https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install "Adafruit SSD1306" "Adafruit GFX Library" \
  "Adafruit BusIO" ArduinoJson "OV5640 Auto Focus for ESP32 Camera"

# from the repo root:
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32S3:PSRAM=opi \
  firmware/lstk_eye_glasses
arduino-cli upload -p /dev/cu.usbmodemXXXX \
  --fqbn esp32:esp32:XIAO_ESP32S3:PSRAM=opi firmware/lstk_eye_glasses
arduino-cli monitor -p /dev/cu.usbmodemXXXX -c baudrate=115200
```

## Gotchas

| Symptom | Cause / fix |
| --- | --- |
| No serial port appears | Charge-only USB-C cable; use a data-capable one |
| Still no port | Hold the tiny B button while plugging in USB, release, retry upload |
| WiFi "works" only within half a meter | u.FL antenna (small black rectangle) not clicked onto its snap connector |
| Network never found | ESP32 sees 2.4 GHz networks only |
| Garbage in serial monitor | Set monitor to 115200 baud |
| Board runs hot during a session | Normal at preview-streaming duty cycle (Seeed ships this camera with a heatsink); idle firmware runs cool |
| Camera init fails | Tools > PSRAM must be "OPI PSRAM" |
| Photos out of focus | OV5640 autofocus firmware upload failed at boot; check serial log for `[cam]` lines and power-cycle |
| HUD readable directly but mirrored through the optics | Set `HUD_MIRRORED 1` in `device_config.h` (0 is for bench use without the beamsplitter) |

## Behavior summary

- **Single click** - capture an XGA JPEG, `POST /api/v1/photos`, show the
  returned scene (falls back to a local counter screen if the server is
  unreachable).
- **Long press** - record the mic while held (20 s cap, `REC` screen); on
  release the WAV goes to `POST /api/v1/ask` and the first slide appears.
- **Double click** - `POST /api/v1/event {"type":"next"}`: next slide.
- **Session active** - a QVGA preview frame goes to `POST /api/v1/preview`
  every 400 ms; returned scenes update the HUD (redraw only when `seq`
  changes); a response with `active=false` ends the session.

HTTP requests are blocking by design; button input is not sampled during an
in-flight request (~100-300 ms for previews, seconds for `/ask`). Click
between preview beats, not during the `thinking...` screen.
