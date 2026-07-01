# tts.py
# Speak the reply with a natural neural voice (edge-tts, free, no key, no local model).
# Synthesizes to a temp mp3 and plays it with pygame. Prints as a fallback.
import asyncio
import os
import tempfile

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

VOICE = "en-US-AriaNeural"
_MP3 = os.path.join(tempfile.gettempdir(), "tacet_tts.mp3")


def speak(text: str) -> None:
    if not text:
        return
    try:
        asyncio.run(_synth(text))
        _play()
    except Exception as e:
        print(f"[tts:{e}] {text}")


async def _synth(text: str) -> None:
    import edge_tts
    await edge_tts.Communicate(text, VOICE).save(_MP3)


def _play() -> None:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(_MP3)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    pygame.mixer.music.unload()          # free the file so the next turn can overwrite it
