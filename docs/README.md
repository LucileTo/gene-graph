# Documentation

Read in this order:

| File | Read it when you want to know... |
|---|---|
| [glossary.md](glossary.md) | what a word means (ontology, mutation, pathway, biomarker, MeSH, PMID, ...) |
| [pipeline-steps.md](pipeline-steps.md) | what each of the three scripts does and how they differ |
| [data-files.md](data-files.md) | what every field of `abstract.json`, `keywords.json` and `association.json` contains |

## The pipeline in one picture

```
PubMed ──► src/pubmed_extraction.py ──► data/abstract.json     (articles: title, abstract, year...)
                                              │
PubTator3 ► src/keyword_builder.py ───────────┴─► data/keywords.json   (genes / diseases / drugs found in each abstract)
                                                        │
LLM ──────► src/associations_llm.py ────────────────────┴─► Neo4j  (relations = the graph edges)
                                                                  │
                                                      app.py (Streamlit) draws the graph
```

`data/` is git-ignored: every teammate generates their own copy by running the three scripts.
