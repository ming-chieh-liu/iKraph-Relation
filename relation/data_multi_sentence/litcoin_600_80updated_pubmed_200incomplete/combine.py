import json
from pathlib import Path

PUBMED_DIR = Path(
    "/data/mliu/iKraph/relation/data_multi_sentence/pubmed_200incomplete/"
    "multi_sentence_split_pubmed_200incomplete"
)

LITCOIN_DIR = Path(
    "/data/mliu/iKraph/relation/data_multi_sentence/litcoin_600_80updated/"
    "multi_sentence_split_litcoin_600_80updated"
)

OUT_DIR = Path("./multi_sentence_split_litcoin_600_80updated_pubmed_200incomplete")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SPLITS = 5
DEDUP_BY_RELATION_ID = False  # set True if you want to remove duplicates

def load_list(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} is {type(data)}, expected a JSON list.")
    return data

for i in range(NUM_SPLITS):
    lit_path = LITCOIN_DIR / f"split_{i}.json"
    pub_path = PUBMED_DIR / f"split_{i}.json"
    out_path = OUT_DIR / f"split_{i}.json"

    lit = load_list(lit_path)
    pub = load_list(pub_path)

    # Your requirement: new split_i is the combine from two split_i directly
    combined = lit + pub  # if you prefer pubmed first: pub + lit

    if DEDUP_BY_RELATION_ID:
        seen = set()
        deduped = []
        for item in combined:
            rid = item.get("relation_id")
            if rid is None or rid not in seen:
                deduped.append(item)
                if rid is not None:
                    seen.add(rid)
        combined = deduped

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"[split_{i}] litcoin={len(lit)} pubmed={len(pub)} -> combined={len(combined)}  wrote: {out_path}")
