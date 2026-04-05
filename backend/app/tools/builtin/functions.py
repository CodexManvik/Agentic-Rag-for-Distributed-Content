from typing import Any


def word_count(args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text", ""))
    words = [w for w in text.split() if w.strip()]
    return {
        "word_count": len(words),
        "text": text,
    }


def echo(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "echo": args,
    }
