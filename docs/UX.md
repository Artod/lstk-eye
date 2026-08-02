# Interaction spec

The single source of truth for what the wearer sees and what every gesture
does in every state. Code follows this document; when they disagree, one of
them is a bug. Principles:

1. **Every screen names the next action.** No dead ends.
2. **A gesture always does something defined** - nothing falls through into
   another flow.
3. **Transient messages never destroy context**: prompts hold ~4 s, then the
   previous content returns on its own.
4. **Commands are intercepted locally** (server-side keyword match, before
   STT output ever reaches the LLM): "calibrate", "next", "back", "repeat",
   "cancel" and their Russian forms never spend an API call. Only real
   questions go to the model.

## Gestures

| Gesture | Meaning everywhere |
|---|---|
| single click | take a photo (in calibration: capture the aligned point) |
| double click | reset - end whatever is happening, clean slate |
| long press | talk; release sends |

## States and screens

### Boot
`lstk-eye fw x.y.z` -> `WiFi connecting...` -> `server ok` / `server
unreachable`. WiFi drop later: `WiFi down / retrying in bg` once, reconnect
is silent, buttons keep working offline (photos count locally).

### Idle (no chat)
Screen: `ready` (or last `done`). Inputs:
- click -> `[N] / photo saved`
- question -> pipeline -> answer (below)
- voice command (next/back/...) -> `no session / hold btn + ask`
- "calibrate" -> calibration
- double click -> `done`

### Chat answer
The pipeline returns either:
- **find-answer** (single anchored step): object label on top + bracket
  frame around the object; brackets follow the object 1-2 Hz; target lost ->
  `look back` on the status row; target outside the visible window ->
  compass chevron at the border pointing where to turn.
- **steps answer** (several steps): instruction text + arrow + `i/N`
  counter. "next"/"back"/"repeat" by voice; past the last step -> `done`.
- **text answer** (nothing to point at): text only.

While a chat is active: click -> badge `[N]` on the status row (scene kept);
follow-up question with a fresh photo -> new answer, history retained (the
model sees previous turns); follow-up without a photo -> `no photo / click,
then ask` for 4 s, then the answer returns; double click -> `done`, history
cleared.

### Calibration
Enter: say "calibrate" from anywhere (a live chat ends: first status line
says `chat ended`).

Screen: bracket crosshair + `point i/3 - click` + live feedback line
(~2 Hz):

| Feedback | Meaning -> action |
|---|---|
| `marker OK - click` | click counts now |
| `no marker seen` | camera does not see it -> turn head toward it / closer |
| `closer to marker` / `further from marker` | resize/move until OK |
| `marker near edge` | about to leave the camera frame -> re-center |
| `didn't catch that` | you spoke, transcript was empty |
| `busy: 2click exits` | you asked a question mid-calibration |

Clicks on a non-OK line do not count and do not advance. Points: center ->
right -> top, all brackets fully on screen. After the 3rd accepted click:

- success -> **`calibrated / hold btn + ask`** - mapping applied instantly,
  saved to the TOML, ready to use, no restart, no extra step.
- impossible solution -> `calib failed-redo 1` - flow restarts at point 1.

Exits: double click (`calibration off`) or voice "cancel"; "calibrate"
again restarts from point 1. Mirrored solutions (camera rotation config off
by 180) are accepted - calibration still works, a warning lands in the log.

### Errors
Any error screen: `! error` + reason + `2click = reset` - on both server
scenes and firmware-local screens.

## Message vocabulary

All wearer-facing strings live in `SceneComposer` and this table; keep them
<= 14 chars/line (visible width at default pads), ASCII only.

| Screen | Line(s) |
|---|---|
| ready | `ready` |
| photo (idle) | `[N]` + `photo saved` |
| photo (in chat) | badge `[N]` on status row |
| bad photo | `bad photo` + `click again` (badge `[!photo]` in chat) |
| thinking | `thinking...` |
| no photo | `no photo` + `click, then ask` |
| photos expired | `photos expired` + `click, then ask` |
| empty speech | `didn't catch that` + `try again` |
| idle command | `no session` + `hold btn + ask` |
| done | `done` |
| calibration end | `calibrated` + `hold btn + ask` |
| calibration abort | `calibration off` |
| error | `! <reason>` + `2click = reset` |
