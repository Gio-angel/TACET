# preprocess.py
# The one and only text-normalizer, shared by the whole pipeline.
# we lowercase and strip punctuation here
# use everywhere
import re

_PUNCT = re.compile(r"[^\w\s]")   # anything that isn't a letter/digit/space
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = str(text).lower()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text
