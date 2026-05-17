from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod


class Speaker(ABC):
    @abstractmethod
    def say(self, text: str) -> None:
        """Speak or display text back to the user."""


class ConsoleSpeaker(Speaker):
    def say(self, text: str) -> None:
        print(f"reachy> {text}")


class ElevenLabsSpeaker(Speaker):
    """Minimal ElevenLabs text-to-speech adapter.

    This writes MP3 bytes to `output_path`. Playback is intentionally left to the
    caller because hardware audio output differs across Reachy deployments.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        output_path: str = "reachy_response.mp3",
    ):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID")
        self.output_path = output_path
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is required")
        if not self.voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID is required")

    def say(self, text: str) -> None:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        payload = json.dumps(
            {
                "text": text,
                "model_id": "eleven_multilingual_v2",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            audio = response.read()
        with open(self.output_path, "wb") as f:
            f.write(audio)
        print(f"reachy> {text}")

