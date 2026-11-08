"""Drug-repurposing candidates for ALS: molecules acting, in the beneficial direction, on the
mechanisms documented in ALS, found in articles about other neurodegenerative diseases."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import requests

from graph_model import ALS_ID, ALS_RELATION, MECHANISMS

ROOT = Path(__file__).resolve().parent.parent
ASSOCIATION_FILE = ROOT / "data" / "association.json"
OUTPUT_FILE = ROOT / "data" / "candidates.json"

TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"

# Effect of a molecule that would help, for each mechanism:
# boost the protective mechanisms, block the harmful ones
BENEFICIAL = {
    "MECH:autophagy": {"activates", "induces", "agonist_of"},
    "MECH:axonal_transport": {"activates", "induces", "agonist_of"},
    "MECH:oxidative_stress": {"inhibits"},
    "MECH:excitotoxicity": {"inhibits"},
    "MECH:neuroinflammation": {"inhibits"},
}

# Generic molecules and toxins that are not real candidates (lowercase names)
BLOCKLIST = {
    "adenosine triphosphate", "atp", "rna", "dna", "water", "oxygen", "glucose", "calcium",
    "reactive oxygen species", "hydrogen peroxide", "superoxide", "nitric oxide", "glutamate",
    "amino acids", "lipids", "peptides", "proteins",
}

# Drugs already approved or in late clinical use for ALS (lowercase names)
KNOWN_ALS_DRUGS = {
    "riluzole", "edaravone", "tofersen", "4-phenylbutyric acid", "sodium phenylbutyrate",
    "ursodoxicoltaurine", "taurursodiol", "tauroursodeoxycholic acid",
}


# ------------------------------------------------------------------ data

def load(path: Path) -> tuple[list[dict], set[str]]:
    """Return (one row per extracted relation and article, mechanisms linked to ALS)."""
    data = json.loads(path.read_text("utf-8"))
    rows, als_mechanisms = [], set()
    for a in data["associations"]:
        if a["negated"]:
            continue
        if a["relation"] == ALS_RELATION and a["object_id"] == ALS_ID:
            als_mechanisms.add(a["subject_id"])
            continue
        for i, pmid in enumerate(a["pmids"]):
            rows.append({
                "s": a["subject_id"], "s_name": a["subject"], "s_type": a["subject_type"],
                "rel": a["relation"],
                "o": a["object_id"], "o_name": a["object"], "o_type": a["object_type"],
                "pmid": pmid, "year": a["years"][i], "outcome": a["outcomes"][i],
            })
    return rows, als_mechanisms


def is_excluded(name: str) -> bool:
    name = name.strip().lower()
    return name in BLOCKLIST or name in KNOWN_ALS_DRUGS


# ------------------------------------------------------------------ ranking

def rank_candidates(rows: list[dict], als_mechanisms: set[str], min_articles: int = 1) -> list[dict]:
    names = {r["s"]: r["s_name"] for r in rows}
    mechanisms, pmids = defaultdict(dict), defaultdict(set)
    diseases = defaultdict(set)

    for r in rows:
        if r["s_type"] != "Molecule" or is_excluded(r["s_name"]):
            continue
        if r["o"] in als_mechanisms and r["rel"] in BENEFICIAL.get(r["o"], ()):
            mechanisms[r["s"]][r["o"]] = r["rel"]
            pmids[r["s"]].add(r["pmid"])
        elif r["rel"] == "treatment":
            diseases[r["s"]].add(f"{r['o_name']} (treatment)")
        elif r["rel"] == "tested":
            diseases[r["s"]].add(f"{r['o_name']} (tested: {r['outcome'] or 'unknown'})")

    candidates = []
    for molecule, mechs in mechanisms.items():
        if len(pmids[molecule]) < min_articles:
            continue
        # 1 point per ALS mechanism + a little more trust with more articles
        # + 0.5 if the molecule already works (treatment or positive test) in another disease
        works = any("treatment" in d or "positive" in d for d in diseases[molecule])
        score = len(mechs) + 0.25 * min(len(pmids[molecule]) - 1, 4) + 0.5 * works
        candidates.append({
            "molecule_id": molecule, "molecule": names[molecule], "score": round(score, 2),
            "n_articles": len(pmids[molecule]),
            "mechanisms": sorted(f"{rel} {MECHANISMS[m][0]}" for m, rel in mechs.items()),
            "other_diseases": sorted(diseases[molecule]),
            "pmids": sorted(pmids[molecule]),
        })
    candidates.sort(key=lambda c: (-c["score"], -c["n_articles"], c["molecule"]))
    return candidates


def trials_count(name: str) -> int | None:
    """Number of ClinicalTrials.gov studies testing this molecule in ALS (None if the API fails)."""
    try:
        response = requests.get(TRIALS_API, timeout=30, params={
            "query.cond": "amyotrophic lateral sclerosis", "query.intr": name,
            "pageSize": 1, "countTotal": "true"})
        response.raise_for_status()
        return response.json().get("totalCount", 0)
    except (requests.RequestException, ValueError):
        return None


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=ASSOCIATION_FILE, help="association.json")
    parser.add_argument("--min-articles", type=int, default=1, help="Minimum supporting articles")
    parser.add_argument("--trials", action="store_true", help="Count ALS trials on ClinicalTrials.gov")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows, als_mechanisms = load(args.file)
    candidates = rank_candidates(rows, als_mechanisms, args.min_articles)[:args.top]
    if args.trials:
        for c in candidates:
            c["als_trials"] = trials_count(c["molecule"])
            time.sleep(0.3)

    print(f"\n=== Molecules acting on ALS mechanisms ({len(candidates)}) ===")
    for c in candidates:
        extra = f" | ALS trials: {c['als_trials']}" if args.trials else ""
        print(f"{c['score']:>5}  {c['molecule']:<28} {', '.join(c['mechanisms'])} | "
              f"elsewhere: {', '.join(c['other_diseases']) or '-'} | articles: {c['n_articles']}{extra}")

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(candidates, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
