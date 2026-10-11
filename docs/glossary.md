# Glossary

Plain-language definitions, with a programming analogy where one helps.
Examples come from ALS (amyotrophic lateral sclerosis), the topic of this project.

## 1. Knowledge-graph vocabulary

| Term | Meaning |
|---|---|
| **Knowledge graph** | Facts stored as a network: things (nodes) connected by labelled links (edges). |
| **Node / Entity** | One "thing": a gene, a disease, a drug. In the code: an *entity* with an `id`, `type` and `name`. |
| **Edge / Relation** | A labelled, directed link between two nodes, e.g. `SOD1 --ASSOCIATED_WITH--> ALS`. |
| **Triple / Triplet** | One fact = (subject, relation, object). `association.json` calls a single extracted fact a *triplet*. |
| **Ontology** | The list of *allowed kinds* of nodes and edges, and which kind may link to which. Analogy: the **schema / type system** of the graph. Without it an LLM could invent any link it likes. |
| **NER** (named-entity recognition) | Finding names of things in text ("SOD1" is a gene). PubTator does this for us. |
| **Normalization** | Mapping different spellings to one official ID. "SOD1", "superoxide dismutase 1" and "Cu/Zn SOD" all become `NCBIGene:6647`. Analogy: resolving aliases to a primary key. |

### The ontology of *this* project

Three node types and four relation types. The direction matters.

| Relation | From → To | Meaning | Example |
|---|---|---|---|
| `ASSOCIATED_WITH` | Gene → Disease | the gene (mutations, expression, protein) is involved in the disease | SOD1 → ALS |
| `TARGETS` | Drug → Gene | the drug acts on the gene, its RNA or its protein | tofersen → SOD1 |
| `TESTED_FOR` | Drug → Disease | the drug was tested as a treatment; has an *outcome* | riluzole → ALS |
| `INTERACTS_WITH` | Gene → Gene | the two genes or their proteins interact (treated as undirected) | gene A ↔ gene B |

It is defined in two places that must be kept in sync:
`PROMPT` and `RELATION_SIGNATURES` in `src/associations_llm.py`, plus
`RELATION_TYPES` and `COLORS` in `app.py`.
