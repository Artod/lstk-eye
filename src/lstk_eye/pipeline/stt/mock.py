"""Mock STT backend.

Contract shared with the simulator: if the "WAV" bytes start with the ASCII
prefix ``MOCKTEXT:`` the remainder is the transcript, UTF-8 encoded. Anything
else transcribes to a canned question. This lets tests and the simulator
inject arbitrary questions without producing audio.
"""

from lstk_eye.pipeline.interfaces import SpeechToText
from lstk_eye.pipeline.types import Transcript

MOCK_PREFIX = b"MOCKTEXT:"
CANNED_TRANSCRIPT = "how do I check battery voltage"


class MockSpeechToText(SpeechToText):
    def transcribe(self, wav: bytes) -> Transcript:
        if wav.startswith(MOCK_PREFIX):
            return Transcript(text=wav[len(MOCK_PREFIX) :].decode("utf-8"))
        return Transcript(text=CANNED_TRANSCRIPT)
