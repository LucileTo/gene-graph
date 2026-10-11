"""Step 3: extract the relations between the entities of each abstract with an LLM -> Neo4j

    python src/associations_llm.py [--limit 20] [--force]

(:Entity {id, name, type})-[:ASSOCIATED_WITH | TARGETS | TESTED_FOR | INTERACTS_WITH {pmid, evidence, ...}]->(:Entity)
One relationship per (article, relation). Each article's LLM status (extracted / skipped / error) is stored on
an (:Article {pmid, year, status, error, rejected}) node: this is what lets a stopped run resume.
"""

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import openai
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from rapidfuzz import fuzz
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
ABSTRACT_FILE = ROOT / "data" / "abstract.json"
KEYWORDS_FILE = ROOT / "data" / "keywords.json"
load_dotenv(ROOT / ".env")

TEMPERATURE = 0
REQUESTS_PER_MINUTE = 20  # stay below the free-tier limit
MAX_RETRIES = 6           # on rate-limit errors (HTTP 429)
EVIDENCE_MIN_SCORE = 90   # minimum similarity (0-100) between the evidence sentence and the abstract

logger = logging.getLogger("associations_llm")


# ------------------------------------------------------------------ prompt

# Change the version every time you edit the prompt: it is saved with each relation.
PROMPT_VERSION = "v1"
PROMPT = """\
You are a biomedical curator building a knowledge graph about amyotrophic lateral sclerosis (ALS) from PubMed abstracts.

You receive the title and abstract of one article, and a list of entities already recognized in the text (genes, diseases, drugs/chemicals), each with an identifier.

Your task: extract the relations between these entities that are explicitly stated in the text.

ALLOWED RELATIONS (closed list, respect the direction subject -> object):
- ASSOCIATED_WITH: Gene -> Disease. The gene (or its mutations, expression, protein) is involved in the disease.
- TARGETS: Drug -> Gene. The drug acts on the gene, its RNA or its protein (inhibits, activates, binds, reduces, degrades...).
- TESTED_FOR: Drug -> Disease. The drug was tested as a treatment of the disease (clinical trial, animal model, cell model). Give "outcome":
    "positive" = beneficial effect shown; "negative" = no benefit or harmful;
    "inconclusive" = mixed or unclear results; "ongoing" = trial planned or in progress, no result yet.
- INTERACTS_WITH: Gene -> Gene. The two genes or their proteins interact physically or functionally.

STRICT RULES:
1. Use ONLY entities from the provided list. Copy their "id" exactly. Never invent an entity or an identifier.
2. "evidence" must be ONE sentence copied word for word from the title or the abstract, the one that best supports the relation. Do not paraphrase, do not shorten, do not merge sentences.
3. "certainty": "established" if the text states the relation as a finding or a known fact; "suggested" if it is a hypothesis, a hint, a trend or uses words like "may", "might", "could", "suggest", "potential".
4. "negated": true if the text explicitly states that the relation does NOT exist (e.g. "no association was found", "did not improve survival"). Still report these relations. A failed drug trial is TESTED_FOR with outcome "negative" and negated false.
5. Do not report relations that are only background knowledge not stated in this text, side effects, or adverse events.
6. Diseases other than ALS are allowed (e.g. frontotemporal dementia).
7. If there is no relation, return {"relations": []}.

OUTPUT: return only a JSON object, with no text before or after, in this exact format:
{
  "relations": [
    {
      "subject": {"text": "<entity name>", "id": "<entity id>"},
      "relation": "ASSOCIATED_WITH | TARGETS | TESTED_FOR | INTERACTS_WITH",
      "object": {"text": "<entity name>", "id": "<entity id>"},
      "evidence": "<exact sentence>",
      "certainty": "established | suggested",
      "negated": false,
      "outcome": "positive | negative | inconclusive | ongoing | null"
    }
  ]
}
"outcome" is null for every relation except TESTED_FOR."""


# ------------------------------------------------------------------ expected LLM answer (pydantic)

class RelationType(str, Enum):
    ASSOCIATED_WITH = "ASSOCIATED_WITH"  # Gene -> Disease
    TARGETS = "TARGETS"                  # Drug -> Gene
    TESTED_FOR = "TESTED_FOR"            # Drug -> Disease (with outcome)
    INTERACTS_WITH = "INTERACTS_WITH"    # Gene -> Gene


class Certainty(str, Enum):
    ESTABLISHED = "established"
    SUGGESTED = "suggested"


class Outcome(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"
    ONGOING = "ongoing"


# Allowed (subject type, object type) for each relation
RELATION_SIGNATURES: dict[RelationType, tuple[str, str]] = {
    RelationType.ASSOCIATED_WITH: ("Gene", "Disease"),
    RelationType.TARGETS: ("Drug", "Gene"),
    RelationType.TESTED_FOR: ("Drug", "Disease"),
    RelationType.INTERACTS_WITH: ("Gene", "Gene"),
}


class EntityRef(BaseModel):
    text: str = Field(min_length=1)
    id: str = Field(min_length=1, description="Identifier from the provided entity list")


class ExtractedRelation(BaseModel):
    """One relation, as returned by the LLM."""
    model_config = ConfigDict(extra="ignore")

    subject: EntityRef
    relation: RelationType
    object: EntityRef
    evidence: str = Field(min_length=10, description="Exact sentence from the abstract")
    certainty: Certainty
    negated: bool
    outcome: Outcome | None = None

    @field_validator("outcome", mode="before")
    @classmethod
    def null_string_to_none(cls, value):
        # Some models write "null" as a string
        return None if isinstance(value, str) and value.strip().lower() in ("", "null", "none") else value

    @model_validator(mode="after")
    def check_outcome(self):
        if self.relation == RelationType.TESTED_FOR and self.outcome is None:
            raise ValueError("'outcome' is required for TESTED_FOR relations")
        if self.relation != RelationType.TESTED_FOR:
            self.outcome = None  # only meaningful for TESTED_FOR
        return self


class ExtractionResult(BaseModel):
    """Full LLM answer for one abstract."""
    relations: list[ExtractedRelation]


class Triplet(ExtractedRelation):
    """A relation with its provenance."""
    pmid: str
    subject_type: str
    object_type: str
    evidence_score: float
    model: str
    prompt_version: str
    extracted_at: str


# ------------------------------------------------------------------ LLM client

class QuotaExhaustedError(RuntimeError):
    """Daily quota probably reached: stop and rerun later."""


class LLMClient:
    def __init__(self):
        self.model = os.getenv("LLM_MODEL", "").strip()
        # max_retries=0: we handle retries ourselves to log them clearly
        self.client = openai.OpenAI(base_url=os.getenv("LLM_BASE_URL", "").strip(),
                                    api_key=os.getenv("LLM_API_KEY", "").strip(),
                                    max_retries=0, timeout=120)
        self.pause = 60 / REQUESTS_PER_MINUTE
        logger.info("LLM model: %s", self.model)

    def chat(self, messages: list[dict]) -> tuple[str, int]:
        """Return (answer, tokens used)."""
        for attempt in range(MAX_RETRIES + 1):
            time.sleep(self.pause)  # stay below the requests-per-minute limit
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"},
                )
                tokens = response.usage.total_tokens if response.usage else 0
                return response.choices[0].message.content or "", tokens
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as error:
                wait = 10 * 2 ** attempt
                logger.warning("%s, waiting %d s (attempt %d/%d)",
                               type(error).__name__, wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
        raise QuotaExhaustedError(f"Still failing after {MAX_RETRIES} retries.")


# ------------------------------------------------------------------ extraction of one article

def build_user_message(record: dict) -> str:
    entities = [json.dumps({"id": e["id"], "type": e["type"], "name": e["name"],
                            "mentions": sorted({m["text"] for m in e["mentions"]})})
                for e in record["entities"]]
    return (f"TITLE: {record['title']}\n\nABSTRACT: {record['abstract']}\n\n"
            "ENTITIES (use only these ids):\n" + "\n".join(entities))


def has_candidate_pair(entities: list[dict]) -> bool:
    """False if no relation is possible (e.g. only diseases): no LLM call needed."""
    types = Counter(e["type"] for e in entities)
    return any(types[s] >= (2 if s == o else 1) and types[o] >= 1
               for s, o in RELATION_SIGNATURES.values())


def evidence_score(evidence: str, text: str) -> float:
    """0-100: does the evidence sentence appear in the text?"""
    def clean(s: str) -> str:
        s = re.sub(r"[‘’“”]", "'", s.lower())
        s = re.sub(r"[‐-―−]", "-", s)
        return re.sub(r"\s+", " ", s).strip(" .")
    return fuzz.partial_ratio(clean(evidence), clean(text))


def check_relation(relation, entities: dict, text: str, min_score: float):
    """Return (relation, rejection reason or None, evidence score)."""
    score = evidence_score(relation.evidence, text)
    if relation.subject.id not in entities or relation.object.id not in entities:
        return relation, "unknown_entity", score
    if relation.subject.id == relation.object.id:
        return relation, "self_relation", score
    types = (entities[relation.subject.id]["type"], entities[relation.object.id]["type"])
    expected = RELATION_SIGNATURES[relation.relation]
    if types[::-1] == expected and types != expected:
        relation = relation.model_copy(update={"subject": relation.object, "object": relation.subject})
    elif types != expected:
        return relation, "wrong_entity_types", score
    if score < min_score:
        return relation, "evidence_not_found", score
    return relation, None, score


def ask_llm(client: LLMClient, user_message: str) -> ExtractionResult:
    """If the answer is invalid, the error is sent back to the LLM once."""
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": user_message}]
    for attempt in (1, 2):
        answer, _ = client.chat(messages)
        try:
            return ExtractionResult.model_validate_json(answer)
        except ValidationError as error:
            if attempt == 2:
                raise
            messages += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": f"Invalid answer:\n{error}\nReturn the corrected JSON only."}]


def process_article(client: LLMClient, record: dict) -> dict:
    result = ask_llm(client, build_user_message(record))
    entities = {e["id"]: e for e in record["entities"]}
    text = record["title"] + "\n" + record["abstract"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    accepted, rejected = [], []
    for relation in result.relations:
        relation, reason, score = check_relation(relation, entities, text, EVIDENCE_MIN_SCORE)
        if reason:
            rejected.append({**relation.model_dump(mode="json"), "reason": reason})
        else:
            accepted.append(Triplet(
                **relation.model_dump(), pmid=record["pmid"],
                subject_type=entities[relation.subject.id]["type"],
                object_type=entities[relation.object.id]["type"],
                evidence_score=score, model=client.model,
                prompt_version=PROMPT_VERSION, extracted_at=now,
            ).model_dump(mode="json"))
    return {"status": "extracted", "accepted": accepted, "rejected": rejected}


# ------------------------------------------------------------------ Neo4j storage

def connect():
    driver = GraphDatabase.driver(os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                                  auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")))
    driver.verify_connectivity()
    driver.execute_query("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
    driver.execute_query("CREATE CONSTRAINT article_pmid IF NOT EXISTS FOR (a:Article) REQUIRE a.pmid IS UNIQUE")
    return driver


def save_result(driver, article: dict, record: dict, result: dict) -> None:
    """Replace everything stored for this article: its status and its relationships."""
    driver.execute_query("MATCH (:Entity)-[r]->(:Entity) WHERE r.pmid = $pmid DELETE r", pmid=article["pmid"])
    driver.execute_query(
        "MERGE (a:Article {pmid: $pmid}) SET a.year = $year, a.status = $status, a.error = $error, "
        "a.rejected = $rejected", pmid=article["pmid"], year=article["year"], status=result["status"],
        error=result.get("error"), rejected=json.dumps(result["rejected"]))

    entities = {e["id"]: e for e in record["entities"]}
    by_relation = {}
    for t in result["accepted"]:
        subject, obj = t["subject"]["id"], t["object"]["id"]
        if t["relation"] == "INTERACTS_WITH" and subject > obj:
            subject, obj = obj, subject  # A-B and B-A are the same interaction
        props = {k: t[k] for k in ("pmid", "evidence", "certainty", "negated", "outcome",
                                   "evidence_score", "model", "prompt_version", "extracted_at")}
        # the same relation twice in one abstract is stored once (the first one)
        by_relation.setdefault(t["relation"], {}).setdefault((subject, obj, t["negated"]), {
            "subject": {"id": subject, "name": entities[subject]["name"], "type": entities[subject]["type"]},
            "object": {"id": obj, "name": entities[obj]["name"], "type": entities[obj]["type"]},
            "props": props})
    for relation, rows in by_relation.items():
        assert relation in RELATION_SIGNATURES  # relation names cannot be Cypher parameters
        driver.execute_query(
            "UNWIND $rows AS row "
            "MERGE (s:Entity {id: row.subject.id}) ON CREATE SET s.name = row.subject.name, s.type = row.subject.type "
            "MERGE (o:Entity {id: row.object.id}) ON CREATE SET o.name = row.object.name, o.type = row.object.type "
            f"CREATE (s)-[r:{relation}]->(o) SET r = row.props", rows=list(rows.values()))


def run(limit: int | None = None, force: bool = False) -> None:
    if not KEYWORDS_FILE.exists():
        raise SystemExit("No entities found: run src/keyword_builder.py first.")
    articles = json.loads(ABSTRACT_FILE.read_text("utf-8"))[:limit]
    keywords = json.loads(KEYWORDS_FILE.read_text("utf-8"))
    driver = connect()
    if force:
        driver.execute_query("MATCH (n) DETACH DELETE n")
    done, _, _ = driver.execute_query(
        "MATCH (a:Article) WHERE a.status IN ['extracted', 'skipped'] RETURN a.pmid AS pmid")
    done = {r["pmid"] for r in done}

    # Articles that failed last time (status "error") are retried
    todo = [a for a in articles if a["pmid"] in keywords and a["pmid"] not in done]
    logger.info("%d articles to process (prompt version %s).", len(todo), PROMPT_VERSION)
    if todo and not os.getenv("LLM_API_KEY", "").strip():
        logger.warning("LLM_API_KEY is empty in .env: extraction skipped. Get a free key at "
                       "https://console.groq.com/keys and paste it in .env.")
        todo = []
    client = LLMClient() if todo else None

    for article in tqdm(todo, desc="LLM", unit="article"):
        record = keywords[article["pmid"]]
        if not has_candidate_pair(record["entities"]):
            result = {"status": "skipped", "accepted": [], "rejected": []}
        else:
            try:
                result = process_article(client, record)
            except QuotaExhaustedError as error:
                logger.error("Stopping: %s Processed articles are saved, rerun later.", error)
                break
            except ValidationError as error:
                result = {"status": "error", "error": str(error), "accepted": [], "rejected": []}
        save_result(driver, article, record, result)  # a stopped run loses nothing
    driver.close()
    logger.info("Relations written to Neo4j.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="Max number of articles")
    parser.add_argument("--force", action="store_true", help="Redo the LLM extraction of all articles")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(limit=args.limit, force=args.force)
