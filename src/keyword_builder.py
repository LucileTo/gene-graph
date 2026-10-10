"""Step 2: find the genes, diseases and drugs of each abstract with PubTator3 -> data/keywords.json

    python src/keyword_builder.py [--limit 20] [--force]
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
ABSTRACT_FILE = ROOT / "data" / "abstract.json"
KEYWORDS_FILE = ROOT / "data" / "keywords.json"

API_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
BATCH_SIZE = 100
# PubTator3 type -> label in our graph (Species, CellLine... are ignored)
ENTITY_TYPES = {"Gene": "Gene", "Disease": "Disease", "Chemical": "Drug"}

logger = logging.getLogger("keyword_builder")


def make_id(raw_id: str, pubtator_type: str, text: str) -> str:
    """NCBIGene:6647, MESH:D000690, or TEXT:<name> when PubTator3 gives no id."""
    if raw_id in ("", "-", "None"):
        return "TEXT:" + re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if pubtator_type == "Gene":
        return f"NCBIGene:{raw_id}"
    return raw_id  # already prefixed, e.g. MESH:D000690


def parse_document(doc: dict) -> dict:
    texts = {"title": "", "abstract": ""}
    entities: dict[str, dict] = {}

    for passage in doc["passages"]:
        if passage["infons"].get("type") in texts:
            texts[passage["infons"]["type"]] = passage["text"]

        for annotation in passage["annotations"]:
            infons = annotation["infons"]
            if infons.get("type") not in ENTITY_TYPES:
                continue
            # A mention can be normalized to several ids ("6647;6648")
            for raw_id in str(infons.get("identifier") or "-").split(";"):
                entity_id = make_id(raw_id.strip(), infons["type"], annotation["text"])
                entity = entities.setdefault(entity_id, {
                    "id": entity_id,
                    "type": ENTITY_TYPES[infons["type"]],
                    "name": infons.get("name") or annotation["text"],
                    "normalized": not entity_id.startswith("TEXT:"),
                    "mentions": [],
                })
                for loc in annotation["locations"]:
                    entity["mentions"].append(
                        {"text": annotation["text"], "offset": loc["offset"], "length": loc["length"]})

    merge_unnormalized(entities)

    # Kept for a later comparison with the LLM
    relations = []
    for relation in doc.get("relations", []):
        role1, role2 = relation["infons"].get("role1"), relation["infons"].get("role2")
        if isinstance(role1, dict) and isinstance(role2, dict):
            relations.append({
                "type": relation["infons"].get("type"),
                "subject_id": make_id(str(role1.get("identifier")), role1.get("type"), ""),
                "object_id": make_id(str(role2.get("identifier")), role2.get("type"), ""),
            })

    return {"pmid": str(doc["id"]), **texts,
            "entities": list(entities.values()), "pubtator_relations": relations}


def merge_unnormalized(entities: dict[str, dict]) -> None:
    """E.g. TEXT:antisense_oligonucleotide_tofersen -> tofersen (MESH:C000709090)."""
    for text_id in [i for i in entities if i.startswith("TEXT:")]:
        texts = {m["text"].lower() for m in entities[text_id]["mentions"]}
        candidates = []
        for other in entities.values():
            if other["normalized"] and other["type"] == entities[text_id]["type"]:
                names = {m["text"].lower() for m in other["mentions"]} | {other["name"].lower()}
                if any(re.search(rf"\b{re.escape(n)}$", t) for n in names for t in texts):
                    candidates.append(other)
        if len(candidates) == 1:  # merge only if there is no ambiguity
            candidates[0]["mentions"] += entities.pop(text_id)["mentions"]


def fetch_batch(pmids: list[str]) -> list[dict]:
    """Returns only the PMIDs annotated by PubTator3."""
    for attempt in range(5):
        try:
            response = requests.get(API_URL, params={"pmids": ",".join(pmids)}, timeout=60)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            data = response.json()
            # If none of the PMIDs is annotated: {"detail": "Could not retrieve publications"}
            return data.get("PubTator3", [])
        except (requests.RequestException, ValueError) as error:
            logger.warning("PubTator3 error (%s), retrying in %d s", error, 5 * 2 ** attempt)
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError("PubTator3 still failing after 5 attempts")


def run(limit: int | None = None, force: bool = False) -> None:
    if not ABSTRACT_FILE.exists():
        raise SystemExit("No articles found: run src/pubmed_extraction.py first.")
    pmids = [a["pmid"] for a in json.loads(ABSTRACT_FILE.read_text("utf-8"))][:limit]
    keywords = {} if force or not KEYWORDS_FILE.exists() else json.loads(KEYWORDS_FILE.read_text("utf-8"))

    to_fetch = [p for p in pmids if p not in keywords]
    logger.info("%d articles, %d already annotated, %d to fetch.",
                len(pmids), len(pmids) - len(to_fetch), len(to_fetch))

    received = set()
    for start in tqdm(range(0, len(to_fetch), BATCH_SIZE), desc="PubTator3", unit="batch"):
        for doc in fetch_batch(to_fetch[start:start + BATCH_SIZE]):
            record = parse_document(doc)
            keywords[record["pmid"]] = record
            received.add(record["pmid"])
        KEYWORDS_FILE.write_text(json.dumps(keywords, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(0.4)  # max 3 requests per second

    # Recent articles may not be in PubTator3 yet: retried at the next run
    missing = [p for p in to_fetch if p not in received]
    logger.info("Annotated %d articles, saved in %s", len(received), KEYWORDS_FILE)
    if missing:
        logger.warning("%d PMIDs not (yet) in PubTator3, will be retried next run: %s",
                       len(missing), ", ".join(missing[:10]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="Max number of articles")
    parser.add_argument("--force", action="store_true", help="Annotate all articles again")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    run(limit=args.limit, force=args.force)
