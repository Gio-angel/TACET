# llm.py
# Turn the finished transcript into a short spoken reply via Gemini.
# Key comes from environment / .env. Falls back to a spoken message if it's missing.
import os

try:
    from dotenv import load_dotenv
    load_dotenv()                      # load key from a local .env if present
except Exception:
    pass

SYSTEM = "You are a helpful voice assistant. Answer in one or two short spoken sentences."
GEMINI_MODEL = "gemini-2.5-flash"


def reply(text: str, model: str = "Gemini") -> str:
    if not text:
        return ""
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        m = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM)
        return m.generate_content(text).text.strip()
    except Exception:
        return "sorry, the gemini api is not available right now"
