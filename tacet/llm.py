# llm.py
# LLM reply. Turns the finished transcript into a reply.


def reply(text: str, model: str = "ChatGPT") -> str:
    return f"[{model} mock reply to]: {text}"
