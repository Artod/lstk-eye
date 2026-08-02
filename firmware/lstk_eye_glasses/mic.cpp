#include "mic.h"

#include "ESP_I2S.h"
#include "device_config.h"
#include "pins.h"

namespace {

constexpr uint32_t kSampleRate = 16000;
constexpr uint32_t kBytesPerSecond = kSampleRate * 2;  // 16-bit mono
constexpr size_t kWavHeaderBytes = 44;
constexpr size_t kMaxPcmBytes = (size_t)kBytesPerSecond * REC_MAX_SECONDS;
// One drain chunk = 16 ms of audio; keeps loop() responsive for the button.
constexpr size_t kChunkBytes = 512;

I2SClass i2s;
uint8_t* wav_buf = nullptr;  // header + PCM, PSRAM
size_t pcm_len = 0;
bool recording = false;

void put_u16(uint8_t* p, uint16_t v) {
  p[0] = v & 0xFF;
  p[1] = v >> 8;
}

void put_u32(uint8_t* p, uint32_t v) {
  p[0] = v & 0xFF;
  p[1] = (v >> 8) & 0xFF;
  p[2] = (v >> 16) & 0xFF;
  p[3] = (v >> 24) & 0xFF;
}

// Standard 44-byte PCM WAV header.
void write_wav_header(uint8_t* h, uint32_t pcm_bytes) {
  memcpy(h + 0, "RIFF", 4);
  put_u32(h + 4, 36 + pcm_bytes);
  memcpy(h + 8, "WAVE", 4);
  memcpy(h + 12, "fmt ", 4);
  put_u32(h + 16, 16);  // fmt chunk size
  put_u16(h + 20, 1);   // PCM
  put_u16(h + 22, 1);   // mono
  put_u32(h + 24, kSampleRate);
  put_u32(h + 28, kBytesPerSecond);
  put_u16(h + 32, 2);   // block align
  put_u16(h + 34, 16);  // bits per sample
  memcpy(h + 36, "data", 4);
  put_u32(h + 40, pcm_bytes);
}

}  // namespace

bool mic_begin() {
  // Probe PSRAM availability once; the real buffer is allocated per
  // recording (ownership transfers to the network worker, which frees it
  // after the upload - a reused static buffer would race a second recording
  // against an in-flight upload, and freeing a persistent buffer corrupts
  // the heap, both observed the hard way).
  void* probe = ps_malloc(kWavHeaderBytes + kMaxPcmBytes);
  if (probe == nullptr) {
    return false;
  }
  free(probe);
  i2s.setPinsPdmRx(PIN_PDM_CLK, PIN_PDM_DATA);
  return true;
}

bool mic_record_start() {
  if (recording) {
    return false;
  }
  wav_buf = (uint8_t*)ps_malloc(kWavHeaderBytes + kMaxPcmBytes);
  if (wav_buf == nullptr) {
    return false;
  }
  if (!i2s.begin(I2S_MODE_PDM_RX, kSampleRate, I2S_DATA_BIT_WIDTH_16BIT,
                 I2S_SLOT_MODE_MONO)) {
    free(wav_buf);
    wav_buf = nullptr;
    return false;
  }
  pcm_len = 0;
  recording = true;
  return true;
}

void mic_record_poll() {
  if (!recording) {
    return;
  }
  const size_t room = kMaxPcmBytes - pcm_len;
  if (room == 0) {
    return;  // cap reached; keep the channel running until release
  }
  const size_t want = room < kChunkBytes ? room : kChunkBytes;
  const size_t got =
      i2s.readBytes((char*)(wav_buf + kWavHeaderBytes + pcm_len), want);
  pcm_len += got;
}

bool mic_record_full() {
  return recording && pcm_len >= kMaxPcmBytes;
}

uint32_t mic_record_seconds() {
  return pcm_len / kBytesPerSecond;
}

size_t mic_record_stop(uint8_t** out_wav) {
  if (!recording) {
    return 0;
  }
  recording = false;
  i2s.end();
  if (pcm_len == 0) {
    free(wav_buf);
    wav_buf = nullptr;
    return 0;
  }
  write_wav_header(wav_buf, pcm_len);
  // Ownership transfer: the caller (network worker) frees this buffer.
  *out_wav = wav_buf;
  const size_t total = kWavHeaderBytes + pcm_len;
  wav_buf = nullptr;
  return total;
}
