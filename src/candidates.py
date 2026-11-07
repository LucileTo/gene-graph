import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ASSOCIATION_FILE = ROOT / "data" / "association.json"
OUTPUT_FILE = ROOT / "data" / "candidates.json"

ALS_IDS = {"MESH:D000690", "MESH:C531617"}  # ALS and ALS type 1
TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"

# Generic molecules and drug classes that are not real candidates (lowercase names)
BLOCKLIST = {
    "adenosine triphosphate", "atp", "rna", "dna", "rna oligonucleotide", "oligonucleotide",
    "oligonucleotides", "antisense oligonucleotide", "water", "oxygen", "glucose",
    "reactive oxygen species", "amino acids", "lipids", "peptides", "proteins",
}

# Drugs already approved or in late clinical use for ALS (lowercase names)
KNOWN_ALS_DRUGS = {
    "riluzole", "edaravone", "tofersen", "4-phenylbutyric acid", "sodium phenylbutyrate",
    "ursodoxicoltaurine", "taurursodiol", "tauroursodeoxycholic acid",
}

# ------------------------------------------------------------------ data

def flatten(path: Path) -> list[dict]:
    """One row per (relation, article): needed to filter by publication year."""
    data = json.loads(path.read_text("utf-8"))
    rows = []
    for a in data["associations"]:
        if a["negated"]:
            continue
        for i, pmid in enumerate(a["pmids"]):
            rows.append({
                "s": a["subject_id"], "s_name": a["subject"], "s_type": a["subject_type"],
                "rel": a["relation"],
                "o": a["object_id"], "o_name": a["object"], "o_type": a["object_type"],
                "pmid": pmid, "year": a["years"][i], "outcome": a["outcomes"][i],
            })
    return rows


def is_noise(name: str) -> bool:
    return name.strip().lower() in BLOCKLIST


# ------------------------------------------------------------------ ranking

def rank_candidates(rows: list[dict], min_articles: int = 1) -> dict[str, list[dict]]:
    names = {}
    for r in rows:
        names[r["s"]], names[r["o"]] = r["s_name"], r["o_name"]

    als_genes = {r["s"] for r in rows
                 if r["rel"] == "ASSOCIATED_WITH" and r["o"] in ALS_IDS and r["s_type"] == "Gene"}
    tested = [r for r in rows if r["rel"] == "TESTED_FOR" and r["o"] in ALS_IDS and r["s_type"] == "Drug"]
    tested_ids = {r["s"] for r in tested}
    known_ids = {i for i in tested_ids if names[i].lower() in KNOWN_ALS_DRUGS}
    failed_ids = {r["s"] for r in tested if r["outcome"] == "negative"}

    neighbors = defaultdict(set)
    for r in rows:
        if r["rel"] == "INTERACTS_WITH":
            neighbors[r["s"]].add(r["o"])
            neighbors[r["o"]].add(r["s"])

    # ---- 1. target-based candidates
    direct, indirect, pmids = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in rows:
        if r["rel"] != "TARGETS" or r["s_type"] != "Drug":
            continue
        drug, gene = r["s"], r["o"]
        if drug in tested_ids or is_noise(names[drug]):
            continue
        if gene in als_genes:
            direct[drug].add(gene)
            pmids[drug].add(r["pmid"])
        elif neighbors[gene] & als_genes:
            indirect[drug].add(gene)
            pmids[drug].add(r["pmid"])

    def target_score(drug):
        base = len(direct[drug]) + 0.5 * len(indirect[drug])
        bonus = 0.25 * min(len(pmids[drug]) - 1, 4)  # more articles = a little more trust
        return base + bonus

    target_based = [{
        "category": "target-based", "drug_id": d, "drug": names[d],
        "score": round(target_score(d), 2), "n_articles": len(pmids[d]),
        "direct_genes": sorted(names[g] for g in direct[d]),
        "indirect_genes": sorted(names[g] for g in indirect[d]),
    } for d in set(direct) | set(indirect) if len(pmids[d]) >= min_articles]
    target_based.sort(key=lambda c: (-c["score"], -c["n_articles"], c["drug"]))

    # ---- 2. pre-clinical candidates (tested in ALS in the articles, but not a known ALS drug)
    evidence = defaultdict(list)
    for r in tested:
        evidence[r["s"]].append(r)
    preclinical = []
    for drug, items in evidence.items():
        if drug in known_ids or drug in failed_ids or is_noise(names[drug]):
            continue
        articles = {r["pmid"] for r in items}
        if len(articles) < min_articles:
            continue
        positives = sum(r["outcome"] == "positive" for r in items)
        preclinical.append({
            "category": "pre-clinical", "drug_id": drug, "drug": names[drug],
            "score": round(len(articles) + 0.5 * positives, 2), "n_articles": len(articles),
            "outcomes": sorted({r["outcome"] or "unknown" for r in items}),
            "pmids": sorted(articles),
        })
    preclinical.sort(key=lambda c: (-c["score"], c["drug"]))

    return {"target_based": target_based, "preclinical": preclinical}


# ------------------------------------------------------------------ evaluation

def time_split(rows: list[dict], cutoff: int, min_articles: int = 1, ks=(5, 10, 20)) -> None:
    """Graph built with articles <= cutoff; ground truth = drugs tested in ALS in later articles."""
    past = [r for r in rows if r["year"] and r["year"] <= cutoff]
    future = [r for r in rows if r["year"] and r["year"] > cutoff]
    ranked = rank_candidates(past, min_articles)["target_based"]

    tested_before = {r["s"] for r in past if r["rel"] == "TESTED_FOR" and r["o"] in ALS_IDS}
    later = {r["s"] for r in future if r["rel"] == "TESTED_FOR" and r["o"] in ALS_IDS} - tested_before
    pool = {r["s"] for r in past if r["s_type"] == "Drug" and not is_noise(r["s_name"])} - tested_before

    print(f"Cutoff {cutoff}: {len(past)} article-level relations before, {len(future)} after.")
    print(f"Target-based candidates: {len(ranked)} | drugs in pool: {len(pool)} | "
          f"tested in ALS later: {len(later & pool)}")
    if not pool:
        return
    print(f"Baseline (random drug from the pool): {len(later & pool) / len(pool):.1%}")
    for k in ks:
        top = ranked[:k]
        if top:
            hits = sum(c["drug_id"] in later for c in top)
            print(f"  precision@{k}: {hits}/{len(top)} = {hits / len(top):.1%}")


def trials_count(drug_name: str) -> int | None:
    """Number of ClinicalTrials.gov studies testing this drug in ALS (None if the API fails)."""
    try:
        response = requests.get(TRIALS_API, timeout=30, params={
            "query.cond": "amyotrophic lateral sclerosis", "query.intr": drug_name,
            "pageSize": 1, "countTotal": "true"})
        response.raise_for_status()
        return response.json().get("totalCount", 0)
    except (requests.RequestException, ValueError):
        return None


# ------------------------------------------------------------------ main

def print_table(title: str, items: list[dict], trials: bool) -> None:
    print(f"\n=== {title} ({len(items)}) ===")
    for c in items:
        if c["category"] == "target-based":
            detail = (f"direct: {', '.join(c['direct_genes']) or '-'} | "
                      f"indirect: {', '.join(c['indirect_genes']) or '-'}")
        else:
            detail = f"outcomes: {', '.join(c['outcomes'])}"
        extra = f" | ALS trials: {c['als_trials']}" if trials else ""
        print(f"{c['score']:>5}  {c['drug']:<28} {detail} | articles: {c['n_articles']}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=ASSOCIATION_FILE, help="association.json")
    parser.add_argument("--cutoff", type=int, help="Time-split evaluation with this year")
    parser.add_argument("--min-articles", type=int, default=1, help="Minimum supporting articles")
    parser.add_argument("--trials", action="store_true", help="Count ALS trials on ClinicalTrials.gov")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = flatten(args.file)
    if args.cutoff:
        time_split(rows, args.cutoff, args.min_articles)
        return

    result = rank_candidates(rows, args.min_articles)
    result = {key: items[:args.top] for key, items in result.items()}
    for items in result.values():
        for c in items:
            if args.trials:
                c["als_trials"] = trials_count(c["drug"])
                time.sleep(0.3)

    print_table("Pre-clinical in ALS, not a known ALS drug", result["preclinical"], args.trials)
    print_table("Target-based: acts on an ALS gene, not tested in ALS yet", result["target_based"], args.trials)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()