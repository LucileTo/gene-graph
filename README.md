# ALS Knowledge Graph app

hello Daffa & Heru!!!, here a little summary of the project 

```
src/
  pubmed_extraction.py   uses the PubMed API to build data/abstract.json
  keyword_builder.py     uses PubTator to build data/keywords.json
  associations_llm.py    uses the LLM to create the final associations and stores them in Neo4j
data/                    abstract.json and keywords.json
docker-compose.yml       local Neo4j (docker compose up -d, browser on http://localhost:7474)
app.py                   the Streamlit frontend (reads the graph from Neo4j)
.env                     the API keys !should be gitignored on prod
```

Step 3 writes the associations straight to Neo4j, and the app reads them from there.

## How to run


### 0. Prerequisites (skip what you already have)

| Tool | Install | Check |
|---|---|---|
| Git | https://git-scm.com/downloads | `git --version` |
| Python 3.10+ | https://www.python.org/downloads/ (Windows: tick "Add python.exe to PATH") | `python --version` (macOS/Linux: `python3 --version`) |
| Docker | Docker Desktop (Windows/macOS): https://www.docker.com/products/docker-desktop/ ; Linux: Docker Engine + the Compose plugin | `docker compose version` |

You also need two accounts/keys:
- an **LLM API key** from any OpenAI-compatible provider, e.g. Gemini (https://aistudio.google.com/apikey) or Groq (https://console.groq.com/keys). Without it, step 3 of the pipeline is skipped and the graph stays empty.
- an **NCBI account** (email; NCBI asks for it) and an NCBI API key (faster PubMed downloads).

### 1. Get the code

```bash
git clone https://github.com/LucileTo/gene-graph.git
cd gene-graph
```

### 2. Create a Python virtual environment and install the dependencies

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure `.env`

Copy the template (`cp .env.example .env`, or `Copy-Item .env.example .env` in PowerShell), then edit `.env`:
- `NEO4J_PASSWORD`: invent one, at least 8 characters. It is used both to create your local Neo4j and by the scripts. Leave `NEO4J_URI` and `NEO4J_USER` as they are.
- `LLM_API_KEY`: your key. If you use another provider than Gemini, also change `LLM_BASE_URL` and `LLM_MODEL` (examples in the file).
- `NCBI_EMAIL`. 
- `NCBI_API_KEY`.

`.env` contains secrets and is git-ignored: never commit it.

### 4. Start Neo4j

```bash
docker compose up -d
```
The first time, Docker downloads the Neo4j image (a few minutes), then check http://localhost:7474 (login `neo4j` + your `NEO4J_PASSWORD`). The database starts empty: the next step fills it. Note that the password is fixed when the database is first created; if you change `NEO4J_PASSWORD` later, run `docker compose down -v` first (this deletes the data).

### 5. Run the pipeline, in order

`--limit 20` keeps it quick for a first try (a few minutes); drop it for the full corpus (500 articles, much longer).
```bash
python src/pubmed_extraction.py --limit 20   # -> data/abstract.json
python src/keyword_builder.py   --limit 20   # -> data/keywords.json
python src/associations_llm.py  --limit 10   # -> Neo4j
```


### 6. Start the app

```bash
streamlit run app.py
```
It opens on http://localhost:8501. If it says it cannot read Neo4j, check that Docker is running and that `docker compose up -d` was done.


<img width="3842" height="2711" alt="pipeline" src="https://github.com/user-attachments/assets/837989fc-de31-4e4d-b2c8-4bd27dda95af" />
