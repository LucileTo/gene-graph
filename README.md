# Mechanism Knowledge Graph for ALS (gene-graph)

A pipeline that reads scientific abstracts about **neurodegenerative diseases** (Alzheimer, Parkinson,
Huntington…), pulls out "who is linked to whom" around **mechanisms of action**, and stores it as a
**knowledge graph**. The goal is **drug repurposing for ALS** (amyotrophic lateral sclerosis): finding
molecules that act on mechanisms known to matter in ALS, in articles about *other* diseases.

The graph is centered on 5 mechanisms, not on genes. Genes are no longer extracted.


## What is a knowledge graph?

A knowledge graph is a network of **things** (nodes) connected by **relationships** (arrows).

```
 [rapamycin] --induces--> [autophagy] --involved_in--> [Parkinson disease]
  (molecule)              (mechanism)                     (disease)
                               |
                               +--documented_in_literature--> [ALS]   (added by hand)
```

Read as sentences: *"Rapamycin induces autophagy. Autophagy is involved in Parkinson disease.
Autophagy is documented in ALS."* Once many sentences like this are stored together, we can ask:
**which molecules act on mechanisms documented in ALS?**


## The graph model

**Nodes (3 types)**

| Type | What it is | Normalization |
|------|------------|---------------|
| `Mechanism` | Closed list of 5: autophagy, oxidative stress, excitotoxicity, neuroinflammation, axonal transport | Any mechanism in an abstract is mapped to one of the 5 (e.g. *mitophagy* → autophagy, *ROS* → oxidative stress), otherwise ignored |
| `Disease` | Neurodegenerative diseases cited in the abstracts | One node per PubTator identifier (MeSH), named with its most frequent name |
| `Molecule` | Drugs and compounds cited in the abstracts | Same: one node per identifier, synonyms merged |

The list of mechanisms and the relations are defined in one place: `src/graph_model.py`.

**Edges (each one keeps the PMIDs that support it)**

| Relation | Direction | Meaning |
|----------|-----------|---------|
| `involved_in` | Mechanism → Disease | the mechanism takes part in the disease |
| `inhibits` | Molecule → Mechanism | reduces or blocks it (e.g. an antioxidant inhibits oxidative stress) |
| `activates` | Molecule → Mechanism | enhances or restores it |
| `induces` | Molecule → Mechanism | triggers it (e.g. rapamycin induces autophagy, rotenone induces oxidative stress) |
| `agonist_of` | Molecule → Mechanism | agonist of a receptor or pathway of the mechanism |
| `tested` | Molecule → Disease | pre-clinical or clinical study, with an outcome (positive, negative, inconclusive, ongoing) |
| `treatment` | Molecule → Disease | established treatment |
| `documented_in_literature` | Mechanism → ALS | added by hand, `source = "manual"` |

**The ALS node.** ALS is excluded from the PubMed search on purpose, so it never appears in the
abstracts. It is added at the very end, once the graph is built, and linked to the 5 mechanisms by a
`documented_in_literature` edge with `source = "manual"`. Every extracted edge has `source = "llm"`,
so the two are easy to tell apart.


## The pipeline

### Step 1: Download articles (`src/pubmed_extraction.py`)
- **What it does:** searches PubMed (the big database of medical papers) and downloads titles and abstracts.
- **Search symbols:** `AND` = both conditions; `OR` = either one; `NOT` = leave out; `*` = any ending
  (`autophag*` matches *autophagy*, *autophagic*…); `[MeSH]` = official topic label put on each paper;
  `[tiab]` = the word appears in the title or abstract; `hasabstract` = only papers with an abstract.
- **The query** has 3 parts joined by `AND`, minus ALS:
  1. one of the **5 mechanisms** (autophagy, oxidative stress, excitotoxicity, neuroinflammation, axonal transport);
  2. a **drug effect** (inhibitor, activator, inducer, agonist, drug therapy);
  3. a **neurodegenerative disease**;
  
  `NOT` ALS or motor neuron disease (ALS is added by hand at the end, see above).
- **Output:** `data/abstract.json`

### Step 2: Find the important words (`src/keyword_builder.py`)
- **What it does:** sends each abstract to **PubTator**, which highlights diseases and chemicals in the
  text and gives each one a standard ID (e.g. Parkinson disease is always `MESH:D010300`, whatever
  the wording). Genes, species, cell lines are ignored. Chemicals become `Molecule` nodes.
- Mechanisms are not tagged by PubTator: they are handled by the LLM in step 3.
- **Output:** `data/keywords.json`

### Step 3: Find the relationships (`src/associations_llm.py`)
- **What it does:** gives an **LLM** each abstract, its diseases and molecules, and the 5 mechanisms
  (with a description of what each one covers), and asks which of them are connected, and how.
- **Safety checks:** every answer is verified. IDs must come from the list (so an unknown mechanism is
  rejected), the types must fit the relation (a molecule cannot be `involved_in` a disease), and the
  sentence the LLM quotes as proof must really be in the abstract. Bad answers are thrown away.
- **Normalization:** synonyms are merged across the whole corpus (one node per molecule or disease).
- **At the end:** the ALS node and its 5 `documented_in_literature` edges are added.
- **Output:** `data/association.json` (each relation comes with its PMIDs and proof sentences)

### Step 4: Store the graph (`neo4j/query_connection.py`)
- **What it does:** loads `association.json` into **Neo4j**. Each node gets the label `Entity` plus its
  type (`Mechanism`, `Disease` or `Molecule`).
- Use `--reset` to empty the database first (otherwise new data is added to the old one).

### Step 5: Explore it (`app.py`)
- **What it does:** a **Streamlit** web app that reads Neo4j and shows an interactive map of the graph.
  Filter by relation type, search a mechanism, disease or molecule, and click a relation to see the
  exact sentence and the PubMed link that support it. Manual ALS edges are purple and dashed.

### Extra: Candidate ranking (`src/candidates.py`)
- **The question:** *which molecules act, in the helpful direction, on mechanisms documented in ALS?*
- **Helpful direction:** boost the protective mechanisms (`activates` / `induces` / `agonist_of`
  autophagy or axonal transport), block the harmful ones (`inhibits` oxidative stress,
  excitotoxicity or neuroinflammation). So a toxin that *induces* oxidative stress is not a candidate.
- **Score:** 1 point per ALS mechanism the molecule acts on, + 0.25 per extra supporting article
  (max +1), + 0.5 if it is already a `treatment` or `tested` with a positive outcome in another disease.
- **Two lists at the top of the file** (plain Python sets of lowercase names, to be **reviewed by a biologist**):
  - `BLOCKLIST`: generic molecules that are not candidates (ATP, water, glucose, ROS, glutamate…);
  - `KNOWN_ALS_DRUGS`: drugs already used in ALS (riluzole, edaravone, tofersen…), not a discovery.
- `--trials` counts the existing ALS trials of each candidate on ClinicalTrials.gov.
- **Output:** `data/candidates.json` and a printed table.


## How to run

```bash
pip install -r requirements.txt
```

Create a `.env` file (never commit it):

```
NCBI_EMAIL=...
NCBI_API_KEY=...        # optional
LLM_BASE_URL=...        # OpenAI-compatible endpoint, e.g. https://api.groq.com/openai/v1
LLM_API_KEY=...
LLM_MODEL=...           # must support JSON mode
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
```

From the project root:

```bash
python src/pubmed_extraction.py --limit 20   # without --limit: up to 500 articles
python src/keyword_builder.py
python src/associations_llm.py
python neo4j/query_connection.py --reset     # --reset empties Neo4j first
streamlit run app.py
python src/candidates.py                     # ranked candidates
python src/candidates.py --min-articles 2 --trials
```

The scripts resume where they stopped. To start from scratch, move the old files out of `data/`
(or use `--force` on steps 1 to 3).
