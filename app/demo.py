# demo.py
# The live loop that ties everything together. It takes frontend
# (ASR + preprocess + frozen BERT) plus an LLM and a TTS function, and runs the
# turn-taking loop: for each new chunk of speech, get the text and its embedding,
# score it, and let the gate decide whether to answer or keep listening. We pass
# the frontend/LLM/TTS in rather than importing them here, so the head can be
# developed and tested without the heavy ASR+BERT stack.
# Where it fits: the orchestrator that runs the full pipeline end to end.
#
# Frontend interface:
#     frontend.stream()        -> yields audio chunks
#     frontend.process(chunk)  -> (text: str, emb: np.ndarray[768])
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tacet.infer import TacetEndpointer
from tacet.gate import should_respond


def run(frontend, llm, tts, tau=None):
    endpointer = TacetEndpointer(tau=tau)
    transcript = ""
    for chunk in frontend.stream():
        text, emb = frontend.process(chunk)
        transcript = text
        prob = endpointer.probability(emb)
        if should_respond(prob, tau):       # gate: P >= tau ?
            reply = llm(transcript)
            tts(reply)
            transcript = ""                 # reset for the next turn
        # else: keep listening, loop continues


if __name__ == "__main__":
    # from frontend.pipeline import FrontEnd
    # from head.gate import llm, tts
    # run(FrontEnd(), llm, tts)
    print("Wire in FrontEnd plus an LLM and TTS to run live.")
