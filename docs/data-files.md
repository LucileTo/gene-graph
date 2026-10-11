> **Note:** step 3 no longer writes `association.json`; it writes to Neo4j (model described in the docstring of `src/associations_llm.py`). Section 3 below describes the same fields as they were in the JSON.

# Data files

All three live in `data/` (git-ignored) and are produced in order. Every script
reads its own previous output first, so re-running only does the missing work
(`--force` starts over). Samples below were taken from real output and shortened.

---

## 1. `data/abstract.json`

**Made by:** `python src/pubmed_extraction.py [--limit N] [--force]`
**Shape:** a JSON **list**, one object per article. Articles without an abstract are dropped.

```json
{
  "pmid": "39337454",
  "title": "Understanding Amyotrophic Lateral Sclerosis: Pathophysiology, ...",
  "abstract": "This review offers an in-depth examination of ...",
  "year": 2024,
  "journal": "International journal of molecular sciences",
  "mesh_terms": ["Amyotrophic Lateral Sclerosis", "Humans", "Biomarkers", "Genetic Therapy"]
}
```

| Field | Type | Meaning |
|---|---|---|
| `pmid` | string | PubMed ID, the key used by all later files |
| `title`, `abstract` | string | HTML tags removed. Structured abstracts keep labels: `BACKGROUND: ... METHODS: ...` |
| `year` | int or null | publication year (null if PubMed gives none) |
| `journal` | string | journal title |
| `mesh_terms` | list of string | MeSH headings assigned by indexers (see glossary). **Article tags, not entities found in the text.** |

The search itself (query, 2020–2026, max 500, sort by relevance) is set by the constants at the top of `pubmed_extraction.py`.

---

## 2. `data/keywords.json`

**Made by:** `python src/keyword_builder.py [--limit N] [--force]`
**Shape:** a JSON **object** keyed by `pmid`. Articles PubTator hasn't annotated yet are simply missing and are retried next run.

```json
"32203398": {
  "pmid": "32203398",
  "title": "Development of disease-modifying drugs for frontotemporal dementia ...",
  "abstract": "...",
  "entities": [
    {
      "id": "MESH:D057180",
      "type": "Disease",
      "name": "Frontotemporal Dementia",
      "normalized": true,
      "mentions": [{"text": "frontotemporal dementia spectrum disorders", "offset": 43, "length": 42}]
    }
  ],
  "pubtator_relations": [
    {"type": "Association", "subject_id": "MESH:D057180", "object_id": "NCBIGene:8148"}
  ]
}
```

### `entities[]`: what PubTator found in the text

| Field | Meaning |
|---|---|
| `id` | `NCBIGene:<n>` for genes, `MESH:<code>` for diseases and chemicals, or `TEXT:<slug>` when PubTator gave no ID |
| `type` | `Gene`, `Disease` or `Drug`. **`Drug` comes from PubTator's `Chemical`**: it is *any* chemical, see [known-data-issues.md](known-data-issues.md). Species, cell lines and variants are dropped. |
| `name` | PubTator's official name, or the text as written |
| `normalized` | `false` exactly when the id starts with `TEXT:` |
| `mentions[]` | every place the entity appears: `text` as written, `offset` and `length` in characters (counted by PubTator from the start of the title). Only `text` is used later; offsets are stored for reference. |

Two details in `keyword_builder.py`:
- One mention can map to several IDs (`"6647;6648"`); it becomes several entities.
- A `TEXT:` entity is merged into a normalized one of the same type only when exactly one candidate matches the end of its text (e.g. `antisense oligonucleotide tofersen` becomes `tofersen`).

### `pubtator_relations[]`

PubTator's *own* guesses of which entity pairs are related (`type` is e.g. `Association`; no direction information). **Not used yet.** Kept so the LLM's output can be compared with it later.

---

## 3. `data/association.json`

**Made by:** `python src/associations_llm.py [--limit N] [--force]` (needs the `LLM_*` variables in `.env`)
**Shape:** an object with three keys.

```json
{ "stats": {...}, "associations": [...], "articles": {...} }
```

### `stats`: a summary of the run

| Field | Meaning |
|---|---|
| `articles` | articles considered (after `--limit`) |
| `articles_annotated_by_pubtator` | those found in `keywords.json` |
| `articles_sent_to_llm` | those that actually cost an LLM call |
| `entities` | count of distinct entities per type |
| `relations_by_type` | graph edges per relation, **negated ones excluded** |
| `negated_relations` | edges where the text says the relation does not exist |
| `rejected_by_reason` | relations the LLM proposed but our checks refused (see below) |
| `llm_errors` | articles whose answer stayed invalid after one retry; retried next run |

### `associations[]`: the graph edges (what `app.py` reads)

One edge per `(subject, relation, object, negated)`, merged across articles.

```json
{
  "subject_id": "NCBIGene:6647", "subject": "SOD1", "subject_type": "Gene",
  "relation": "ASSOCIATED_WITH",
  "object_id": "MESH:D000690", "object": "Amyotrophic Lateral Sclerosis", "object_type": "Disease",
  "negated": false,
  "n_articles": 1,
  "certainty": "established",
  "outcome": null,
  "model": "gemini-3.5-flash-lite", "prompt_version": "v1", "extracted_at": "2026-10-10T23:47:53+00:00",
  "pmids": ["39330700"],
  "evidences": ["ALS associated with gain of function mutations in SOD1 ..."],
  "outcomes": [null],
  "years": [2024]
}
```

| Field | Meaning |
|---|---|
| `subject_*`, `object_*` | the two ends. `INTERACTS_WITH` is undirected, so its ends are sorted by id. |
| `negated` | the text says the relation does **not** hold |
| `n_articles` | how many different articles support the edge (`= len(pmids)`) |
| `certainty` | `established` if **any** supporting article states it as a finding, else `suggested` |
| `outcome` | only for `TESTED_FOR`: the most frequent outcome across articles (`inconclusive` on a tie), otherwise `null` |
| `model`, `prompt_version`, `extracted_at` | provenance. Several values are joined by `;`. |
| `pmids`, `evidences`, `outcomes`, `years` | **parallel lists**: index *i* of each list describes the same article. This is what the app shows when you select a table row. |

### `articles{}`: raw result per article (lets a run resume)

Keyed by `pmid`. Each value:

```json
{ "status": "extracted", "accepted": [ ...triplets... ], "rejected": [ ...with "reason"... ] }
```

| `status` | Meaning |
|---|---|
| `extracted` | the LLM was called and answered validly |
| `skipped` | no possible pair of entities (e.g. only diseases), so no call was made |
| `error` | the answer was invalid twice; has an `error` message; retried next run |

**`accepted[]` triplet** = the LLM's answer plus provenance:
`subject {text,id}`, `relation`, `object {text,id}`, `evidence`, `certainty`, `negated`, `outcome`,
plus `pmid`, `subject_type`, `object_type`, `evidence_score`, `model`, `prompt_version`, `extracted_at`.

- `evidence_score` is 0–100, the fuzzy similarity between the quoted sentence and the abstract. It must be at least 90 (`EVIDENCE_MIN_SCORE`).

**`rejected[]`** are the LLM's relations that failed a check. `reason` is one of:

| `reason` | What went wrong |
|---|---|
| `unknown_entity` | the LLM used an ID that was not in the entity list (invented) |
| `self_relation` | subject and object are the same entity |
| `wrong_entity_types` | e.g. a Drug → Drug link, which no relation allows. A merely *reversed* pair is silently swapped instead of rejected. |
| `evidence_not_found` | the quoted sentence isn't in the abstract (paraphrased or invented) |

A high count of rejections tells you the prompt or model is weak.
