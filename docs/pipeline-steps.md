# The three pipeline steps

The scripts in `src/` run in order, and each one reads the output of the previous one. Each answers a different question.

| | `pubmed_extraction.py` | `keyword_builder.py` | `associations_llm.py` |
|---|---|---|---|
| **Question** | Which articles are about ALS? | Which genes, diseases and drugs does each article mention? | How are those entities related, according to the text? |
| **Source** | PubMed (Biopython / Entrez) | PubTator3 (NCBI named-entity recognition API) | An LLM (any OpenAI-compatible API) |
| **Input** | A search query (constants in the file) | `abstract.json` | `abstract.json` + `keywords.json` |
| **Output** | `data/abstract.json` | `data/keywords.json` | Neo4j |
| **Content** | PMID, title, abstract, year, journal, MeSH terms | Entities per article (id, type, name, mention texts) | Relations (subject, relation, object) with an evidence sentence |
| **Method** | Plain search and download | Automatic annotation (NER) and ID normalization | LLM extraction, then validation |

## 1. `pubmed_extraction.py`: get the raw material

- Searches for ALS articles from 2020 to 2026 (at most 500, sorted by relevance).
- Downloads titles and abstracts. Articles without an abstract are dropped.
- There is no intelligence here yet: the output is just text.

## 2. `keyword_builder.py`: recognize the entities

- Sends the PMIDs to PubTator3, which tags the genes, diseases and chemicals (chemicals are called `Drug` in our graph) found in the text.
- Every entity gets a standard ID: `NCBIGene:6647` for genes, `MESH:D000690` for diseases and chemicals (that one is ALS). If PubTator gives no ID, we use `TEXT:<slug>`.
- This step only knows **what** is mentioned, not **how things are related**. For example, it knows that `SOD1` and `ALS` appear in the same article, but not whether SOD1 is associated with ALS.

## 3. `associations_llm.py`: decide the relations

- For each article, the LLM receives the abstract and the entity list from step 2. It may only use those entities.
- Only four relations are allowed, each with a fixed direction:
  - `ASSOCIATED_WITH`: Gene → Disease
  - `TARGETS`: Drug → Gene
  - `TESTED_FOR`: Drug → Disease, with an outcome
  - `INTERACTS_WITH`: Gene → Gene
- The answer is checked again: the IDs must be in the entity list, the entity types must fit the relation, and the evidence sentence must really appear in the abstract (no invented quotes).
- Relations that pass are saved to Neo4j, one per article. The status of each article (`extracted`, `skipped` or `error`) is saved too, so a stopped run can resume.
- Articles that cannot contain a relation (for example, only diseases were found) are skipped without calling the LLM, so they cost no quota.

## How they fit together

```
PubMed ──► abstract.json ──► PubTator3 ──► keywords.json ──► LLM ──► Neo4j ──► app.py
          (the articles)     (the entities)                (the relations)  (the graph)
```

The first two steps do not use an LLM and their output can be reused. Only step 3 uses LLM quota and writes to the database. If you change the LLM prompt, you only need to redo step 3 (`--force`, which empties the database).
