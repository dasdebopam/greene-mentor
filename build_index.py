import os, json, numpy as np
from sent_transfom import embed_texts

os.makedirs("index", exist_ok=True)
VEC="index/greene_vectors.npy"; META="index/greene_metadata.jsonl"

texts, meta = [], []
with open("corpus/greene.jsonl", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        txt = o["content"][:1200]
        texts.append(txt)
        meta.append({"title": o["title"], "source": o["source"], "content": txt})

X = embed_texts(texts)             # (N, d) unit-normalized
np.save(VEC, X)
with open(META, "w", encoding="utf-8") as w:
    for m in meta: w.write(json.dumps(m, ensure_ascii=False) + "\n")

print("Built:", X.shape, "->", VEC, "and", META)
