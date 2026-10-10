"""Step 1: download ALS abstracts from PubMed -> data/abstract.json

    python src/pubmed_extraction.py [--limit 20] [--force]
"""

import argparse
import html
import json
import logging
import os
import re
import warnings
from pathlib import Path

from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
ABSTRACT_FILE = ROOT / "data" / "abstract.json"
load_dotenv(ROOT / ".env")

# PubMed search
QUERY = ('"Amyotrophic Lateral Sclerosis"[MeSH] AND (gene* OR genetic* OR therap* OR drug* '
         'OR treatment*) AND hasabstract')
MIN_DATE, MAX_DATE = "2020", "2026"
MAX_ARTICLES = 500
SORT = "relevance"  # or "pub_date" (very recent articles are often not yet in PubTator3)
BATCH_SIZE = 200

logger = logging.getLogger("pubmed_extraction")


def configure_entrez() -> None:
    # Biopython handles the NCBI rate limit and retries by itself
    Entrez.email = os.getenv("NCBI_EMAIL", "").strip() or None
    Entrez.api_key = os.getenv("NCBI_API_KEY", "").strip() or None
    Entrez.max_tries = 5
    if not Entrez.email:
        warnings.filterwarnings("ignore", message=r"\s*Email address is not specified")
        logger.warning("NCBI_EMAIL is empty in .env (recommended by NCBI).")


def parse_article(article: dict) -> dict:
    citation = article["MedlineCitation"]
    art = citation["Article"]

    # Structured abstracts have labelled parts (BACKGROUND, METHODS...)
    parts = []
    for section in art.get("Abstract", {}).get("AbstractText", []):
        label = getattr(section, "attributes", {}).get("Label")
        parts.append(f"{label}: {section}" if label else str(section))

    pub_date = art["Journal"]["JournalIssue"]["PubDate"]
    year = pub_date.get("Year") or str(pub_date.get("MedlineDate", ""))[:4]

    return {
        "pmid": str(citation["PMID"]),
        "title": clean_text(art.get("ArticleTitle", "")),
        "abstract": clean_text("\n".join(parts)),
        "year": int(year) if year.isdigit() else None,
        "journal": str(art["Journal"].get("Title", "")),
        "mesh_terms": [str(h["DescriptorName"]) for h in citation.get("MeshHeadingList", [])],
    }


def clean_text(text) -> str:
    """Remove HTML tags kept by PubMed (<i>SOD1</i> -> SOD1)."""
    return html.unescape(re.sub(r"</?(i|b|u|sub|sup)>", "", str(text))).strip()


def run(limit: int | None = None, force: bool = False) -> None:
    configure_entrez()
    articles = [] if force or not ABSTRACT_FILE.exists() else json.loads(ABSTRACT_FILE.read_text("utf-8"))

    # 1. esearch: list of PMIDs matching the query
    handle = Entrez.esearch(db="pubmed", term=QUERY, mindate=MIN_DATE, maxdate=MAX_DATE,
                            datetype="pdat", retmax=limit or MAX_ARTICLES, sort=SORT)
    result = Entrez.read(handle)
    logger.info("%s articles match the query, keeping %d.", result["Count"], len(result["IdList"]))

    done = {a["pmid"] for a in articles}
    to_fetch = [p for p in result["IdList"] if p not in done]
    logger.info("%d already downloaded, %d to fetch.", len(done & set(result["IdList"])), len(to_fetch))

    # 2. efetch: download the articles by batches
    ABSTRACT_FILE.parent.mkdir(exist_ok=True)
    for start in tqdm(range(0, len(to_fetch), BATCH_SIZE), desc="PubMed", unit="batch"):
        handle = Entrez.efetch(db="pubmed", id=",".join(to_fetch[start:start + BATCH_SIZE]),
                               rettype="abstract", retmode="xml")
        batch = [parse_article(a) for a in Entrez.read(handle)["PubmedArticle"]]
        articles += [a for a in batch if a["abstract"]]
        ABSTRACT_FILE.write_text(json.dumps(articles, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("%d articles saved in %s", len(articles), ABSTRACT_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help=f"Max number of articles (default: {MAX_ARTICLES})")
    parser.add_argument("--force", action="store_true", help="Download all articles again")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    run(limit=args.limit, force=args.force)
