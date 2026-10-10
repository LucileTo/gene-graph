# ALS Knowledge Graph app

hello Daffa & Heru!!!, here a little summary of the project 

```
src/
  pubmed_extraction.py   uses the PubMed API to build data/abstract.json
  keyword_builder.py     uses PubTator to build data/keywords.json
  associations_llm.py    uses the LLM to create the final associations and build data/association.json
data/                    the 3 json files
app.py                   the Streamlit frontend (graph built from data/association.json)
.env                     the API keys !should be gitignored on prod
```

! for now i havent added the neo4j db, this will be the next step, graph is directly build using the assocaitions.json file