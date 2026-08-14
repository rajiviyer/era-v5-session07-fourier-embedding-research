"""Download public-domain English prose and write a medium LM training corpus.

Fetches Project Gutenberg texts (Alice in Wonderland, The Wonderful Wizard of Oz),
strips boilerplate, concatenates, and truncates to a target GPT-2 token count.

    python scripts/fetch_english_corpus.py
    python scripts/fetch_english_corpus.py --max-tokens 80000 --out data/corpus_english.txt
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from byte_table import get_gpt2_tokenizer

# Stable Gutenberg cache URLs (UTF-8 plain text).
SOURCES: tuple[tuple[str, str], ...] = (
    ("Alice's Adventures in Wonderland", "https://www.gutenberg.org/cache/epub/11/pg11.txt"),
    ("The Wonderful Wizard of Oz", "https://www.gutenberg.org/cache/epub/55/pg55.txt"),
    ("The Adventures of Sherlock Holmes (excerpt)", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"),
)

OUT_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "corpus_english.txt"


def _strip_gutenberg_boilerplate(text: str) -> str:
    start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    if start and end:
        return text[start.end() : end.start()].strip()
    return text.strip()


def fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "era-v5-corpus-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def build_corpus(max_tokens: int) -> str:
    tokenizer = get_gpt2_tokenizer()
    parts: list[str] = []

    for title, url in SOURCES:
        print(f"  fetching {title} ...")
        raw = fetch_text(url)
        body = _strip_gutenberg_boilerplate(raw)
        parts.append(f"# {title}\n\n{body}")

    text = "\n\n---\n\n".join(parts)
    ids = tokenizer.encode(text)
    print(f"  raw size {len(ids):,} tokens")
    if len(ids) > max_tokens:
        text = tokenizer.decode(ids[:max_tokens])
        print(f"  truncated to {max_tokens:,} tokens")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch medium English corpus for train.py")
    parser.add_argument("--max-tokens", type=int, default=80_000)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    print(f"Building corpus (target {args.max_tokens:,} GPT-2 tokens)")
    text = build_corpus(args.max_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    n = len(get_gpt2_tokenizer().encode(text))
    print(f"wrote {args.out} ({n:,} tokens, {len(text):,} chars)")


if __name__ == "__main__":
    main()
