# Hardware guide

Everything mounts on the LEFT temple of ordinary prescription glasses. Total parts
cost is roughly 40 USD on top of the glasses themselves. Prices below are indicative;
they drift by supplier and pack size.

## Bill of materials

| Part | Spec | Role | ~Price |
|---|---|---|---|
| Seeed XIAO ESP32S3 Sense | expansion board with camera socket, onboard PDM mic, BAT pads with built-in LiPo charging (~100 mA, active only when the power switch is on) | the only computer and the only radio on the device | $14 |
| OV5640 camera module | 5 MP autofocus, swapped into the Sense camera socket | forward-looking capture and preview frames | $8 |
| SSD1306 OLED | 0.96", 128x64, I2C; active emitting area 21.7 x 10.9 mm | image source for the HUD | $3 |
| Biconvex lens | 25 mm dia / 45 mm focal length (Google Cardboard spec); winged flanges double as glue tabs | collimates the OLED image | $1 |
| Plate beamsplitter | 50/50, cut to ~30 x 25 mm | folds the image into the eye; world light passes through | $8 |
| LiPo cell | 502535, 3.7 V 450 mAh, inline slide switch on BAT+ | power | $4 |
| Push button | big panel-mount, prewired | single/double/long-press gestures | $1 |
| MPU6050 (GY-521), optional | I2C IMU at address 0x68 | planned: 100 Hz gyro shifts the overlay between server anchor updates | $2 |

OV5640 autofocus is dead until firmware is uploaded to the sensor at boot: use the
ESP32_OV5640_AF Arduino library after `esp_camera_init`.

The beamsplitter's coated side faces the lens. A faint second-surface ghost is normal
for a plate splitter.

## Wiring

| Connection | Pins |
|---|---|
| OLED VCC | 3V3 |
| OLED GND | GND |
| OLED SDA | D4 |
| OLED SCL | D5 |
| Button | between D1 and GND, `INPUT_PULLUP`, active-low |
| LiPo | BAT pads, slide switch inline on BAT+ |
| MPU6050 (optional) | same I2C bus as the OLED (D4/D5), address 0x68 |

The two GND pins on the XIAO are the same net - route by convenience. The camera uses
the board socket (many contacts); the display is 4 wires - bandwidth dictates
connectors.

## Optical layout

Top view: the OLED sits at the hinge corner and fires across the front of the glasses.
A lens at distance f collimates the image. A 45-degree beamsplitter in front of the
eye folds the beam back into the pupil; world light passes straight through the same
splitter. The splitter's reflective face must face both the lens and the pupil - its
normal bisects the 90-degree fold. Tilt it the other way and the image goes to the
ceiling.

Perceived result: a virtual image ~27 degrees wide at ~infinity (or ~1.2 m if the OLED
sits ~1 mm inside f - easier on the eyes for desk work), cyan monochrome. The world is
one photographic stop dimmer inside a ~30 x 25 mm patch. The HUD is invisible in
direct sunlight and fine indoors.

The sizing formula:

```
visible screen width  ~ f * (D + p) / L
vignette-free zone    ~ f * (D - p) / L
```

where D is the lens diameter, p ~ 4 mm the pupil, and L the lens-to-eye distance along
the folded path. With f = 45, D = 25, L = 40 mm: visible width ~ 45 * 29 / 40 ~ 33 mm
and vignette-free ~ 45 * 21 / 40 ~ 24 mm - both wider than the 21.7 mm panel, so the
full screen is visible with margin. L is the design currency: every extra centimeter
of air shrinks the window. Pack tight.

The OLED-to-lens-to-splitter channel must be enclosed and matte black inside. This
kills stray reflections of the room and is part of the optical design, not cosmetics.

## Assembly order

Bench-test first, cardboard-prototype second, glue last.

1. Measure the real focal length of every lens sample (focus the sun or a far lamp
   onto paper, measure with a ruler) before designing around it.
2. Bench the optical chain off the glasses: OLED, lens at f, uncut splitter. Verify
   the full screen is visible at the design L.
3. Cut the splitter to ~30 x 25 mm (score and snap), coated side toward the lens.
4. Cardboard-prototype the optics rail on the actual glasses; iterate L and angles
   until the window sits where the eye is.
5. Hot-glue the final build. Hot glue is the assembly method: dielectric, reversible
   with isopropyl worked under the edge. Keep it off chip cans and off optical
   centers. Glue the lens by its flange tabs and the splitter by extreme edges on the
   uncoated side.
6. Placement on the temple: optics rail along the top of the left lens; MCU and camera
   just behind the hinge (camera looks forward, above the splitter, lens proud of the
   glass edge); button on top of the temple; battery and switch at the ear as
   counterweight.

## Hard-won rules

| Rule | Reason |
|---|---|
| Measure every lens's real focal length before it enters a design | listing specs are unreliable |
| Verify LiPo polarity with a meter before it touches the BAT pads | no reverse protection on the board; cheap cells ship reversed |
| Cut battery wires one at a time | a bare cell shorts through the cutter |
| Click the WiFi antenna on | without it WiFi works at half-meter range and mimics a software bug |
| Use a 2.4 GHz network | ESP32 does not see 5 GHz |
| Mirror the SSD1306 image in panel init (SEGREMAP/COMSCAN) | the single reflection mirrors the image; do not fix it optically |
| Expect heat while streaming | normal for this camera at demo duty cycle; battery placement away from the hot chip is deliberate |
