// Onboard PDM microphone: hold-to-talk recording into PSRAM, WAV-wrapped.
//
// 16 kHz, 16-bit, mono; capped at REC_MAX_SECONDS. The I2S channel is started
// on record start and stopped on record stop so no stale audio from between
// recordings leaks into the next request.
#pragma once

#include <Arduino.h>

// Allocate the PSRAM buffer and configure the PDM pins.
bool mic_begin();

// Start a recording. Returns false if one is already running or init failed.
bool mic_record_start();

// Drain the I2S DMA into the buffer; call every loop() while recording.
void mic_record_poll();

// Cap reached: samples are being dropped, release the button.
bool mic_record_full();

uint32_t mic_record_seconds();

// Stop recording, write the 44-byte WAV header, return the complete WAV via
// *out_wav (owned by this module, valid until the next mic_record_start).
// Returns the total byte length, 0 when nothing was recorded.
size_t mic_record_stop(uint8_t** out_wav);
