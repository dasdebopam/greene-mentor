# normalize_corpus_from_llamaparse.py
import os, json, re, pathlib

INPUTS = [
    ("48_laws_transcript.json", "48laws"),
    ("Human_nature_transcripts.json", "human_nature"),
    ("Mastery_transcript.json", "mastery"),
]
os.makedirs("corpus", exist_ok=True)
OUT = "corpus/greene.jsonl"

def clean(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

with open(OUT, "w", encoding="utf-8") as w:
    total = 0
    for fname, book in INPUTS:
        data = json.load(open(fname, encoding="utf-8"))
        pages = data.get("pages", [])
        for p in pages:
            # prefer markdown if present, else plain text
            content = p.get("md") or p.get("text") or ""
            content = clean(content)
            if not content:
                continue
            title = f"{book.capitalize()} — Page {p.get('page', 'N/A')}"
            line = {
                "title": title,
                "content": content[:3000],        # keep chunks compact
                "source": pathlib.Path(fname).stem,
                "meta": {"book": book, "page": p.get("page")}
            }
            w.write(json.dumps(line, ensure_ascii=False) + "\n")
            total += 1

print(f"Wrote {total} chunks -> {OUT}")
