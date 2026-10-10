"""Streamlit app to explore the ALS knowledge graph built from data/association.json

    streamlit run app.py
"""

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st
from pyvis.network import Network

ASSOCIATION_FILE = Path(__file__).resolve().parent / "data" / "association.json"
ALS_ID = "MESH:D000690"
COLORS = {"Gene": "#2a78d6", "Drug": "#eb6834", "Disease": "#1baf7a"}
RELATION_TYPES = ["TARGETS", "INTERACTS_WITH", "ASSOCIATED_WITH", "TESTED_FOR"]
MAX_EDGES = 200  # keep the whole-network view readable

st.set_page_config(page_title="ALS Knowledge Graph", page_icon="🧬", layout="wide")


@st.cache_data
def load_data(path: Path, modified: float):  # `modified`: reload when the file changes
    data = json.loads(path.read_text(encoding="utf-8"))
    relations = pd.DataFrame(data["associations"], columns=[
        "subject_id", "subject", "subject_type", "relation", "object_id", "object", "object_type",
        "negated", "n_articles", "certainty", "outcome", "pmids", "evidences", "outcomes", "years"])
    relations = relations[~relations["negated"].astype(bool)].copy()
    relations["with_als"] = relations["object_id"] == ALS_ID
    return relations, data["stats"]


def als_profile(als_relations: pd.DataFrame) -> dict:
    return {r["subject_id"]: {"relation": r["relation"], "n_articles": r["n_articles"],
                              "outcome": r["outcome"] if isinstance(r["outcome"], str) else None}
            for _, r in als_relations.iterrows()}


def describe_als(profile: dict | None) -> str:
    if not profile:
        return "no direct link with ALS in the corpus"
    if profile["relation"] == "TESTED_FOR":
        return f"tested in ALS ({profile['n_articles']} articles, outcome: {profile['outcome']})"
    return f"associated with ALS ({profile['n_articles']} articles)"


def neighborhood(edges: pd.DataFrame, center: str, depth: int) -> pd.DataFrame:
    reached = {center}
    for _ in range(depth):
        touching = edges["subject_id"].isin(reached) | edges["object_id"].isin(reached)
        reached |= set(edges.loc[touching, "subject_id"]) | set(edges.loc[touching, "object_id"])
    return edges[edges["subject_id"].isin(reached) & edges["object_id"].isin(reached)]


def draw_graph(edges: pd.DataFrame, profiles: dict, center: str | None) -> str:
    net = Network(height="620px", width="100%", directed=True, bgcolor="#ffffff",
                  font_color="#0b0b0b", cdn_resources="in_line")
    net.set_options("""{
      "nodes": {"scaling": {"label": {"drawThreshold": 1}}},
      "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}, "smooth": false},
      "physics": {"barnesHut": {"gravitationalConstant": -2500, "centralGravity": 0.6,
                                "springLength": 90, "avoidOverlap": 0.5},
                  "stabilization": {"iterations": 300}},
      "interaction": {"hover": true}
    }""")
    for _, r in edges.iterrows():
        for node_id, name, label in ((r["subject_id"], r["subject"], r["subject_type"]),
                                     (r["object_id"], r["object"], r["object_type"])):
            profile = profiles.get(node_id)
            n_als = profile["n_articles"] if profile else 0
            tested = bool(profile) and profile["relation"] == "TESTED_FOR"
            net.add_node(node_id, label=name, color={"background": COLORS[label], "border": "#0b0b0b" if tested else COLORS[label]},
                         borderWidth=4 if tested else 1, size=min(10 + 4 * math.sqrt(n_als), 32),
                         title=f"{name} ({label}): {describe_als(profile)}",
                         font={"size": 26 if node_id == center else 20})
        net.add_edge(r["subject_id"], r["object_id"],
                     title=f"{r['relation']} | {r['n_articles']} article(s) | {r['certainty']}",
                     width=1 + 2 * math.sqrt(r["n_articles"]), color="#a3a29c",
                     dashes=r["certainty"] == "suggested")
    return net.generate_html()


def show_evidence(table: pd.DataFrame, key: str) -> None:
    columns = ["subject", "relation", "object", "n_articles", "certainty", "outcome"]
    selection = st.dataframe(table[columns], hide_index=True, width="stretch", key=key,
                             on_select="rerun", selection_mode="single-row")
    if selection.selection.rows:
        row = table.iloc[selection.selection.rows[0]]
        st.markdown(f"#### {row['subject']} → **{row['relation']}** → {row['object']}")
        for pmid, evidence, year, outcome in zip(row["pmids"], row["evidences"], row["years"], row["outcomes"]):
            extra = f" — outcome: **{outcome}**" if outcome else ""
            st.markdown(f"- [PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) ({year}){extra}\n\n  > {evidence}")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🧬 ALS Knowledge Graph")
page = st.sidebar.radio("Page", ["Explorer", "Statistics"])

if not ASSOCIATION_FILE.exists():
    st.warning("No data yet. Run the 3 scripts of `src/` in order "
               "(`python src/pubmed_extraction.py --limit 20`, then keyword_builder, then associations_llm).")
    st.stop()
relations, stats = load_data(ASSOCIATION_FILE, ASSOCIATION_FILE.stat().st_mtime)
if relations.empty:
    st.info("No relations yet: the LLM extraction has not been run. Add your LLM key "
            "in `.env` and run `python src/associations_llm.py`.")
    st.stop()

st.sidebar.header("Filters")
types = st.sidebar.multiselect("Relation types", RELATION_TYPES, default=RELATION_TYPES)
certainties = st.sidebar.multiselect("Certainty", ["established", "suggested"],
                                     default=["established", "suggested"])
min_articles = st.sidebar.slider("Minimum number of articles", 1,
                                 max(int(relations["n_articles"].max()), 2), 1)
years = st.sidebar.slider("Publication years", 2015, 2026, (2020, 2026))

filtered = relations[
    relations["relation"].isin(types)
    & relations["certainty"].isin(certainties)
    & (relations["n_articles"] >= min_articles)
    & relations["years"].apply(lambda ys: any(y and years[0] <= y <= years[1] for y in ys))
]
# ALS is not drawn (the whole corpus is about ALS): links with ALS become node size and tooltip
als_relations = filtered[filtered["with_als"]]
network = filtered[~filtered["with_als"]]
profiles = als_profile(als_relations)

# Every entity of the graph: id -> "name (type)"
label_of = {}
for side in ("subject", "object"):
    label_of.update(zip(relations[f"{side}_id"], relations[side] + " (" + relations[f"{side}_type"] + ")"))

# ------------------------------------------------------------------ Explorer
if page == "Explorer":
    st.title("Explorer")
    st.caption("ALS is not drawn (the whole corpus is about ALS): an entity's link with ALS is "
               "shown by its size and in its tooltip.")

    in_network = sorted(set(network["subject_id"]) | set(network["object_id"]),
                        key=lambda i: label_of[i].lower())
    left, right = st.columns([3, 1])
    center = left.selectbox("Search an entity (gene, drug, disease)", [None] + in_network,
                            format_func=lambda i: "— whole network —" if i is None else label_of[i])
    depth = right.radio("Neighbors", [1, 2], horizontal=True, disabled=center is None,
                        help="2 = also the neighbors of the neighbors (e.g. drug → target → ALS gene)")

    view = neighborhood(network, center, depth) if center else network.nlargest(MAX_EDGES, "n_articles")
    if view.empty:
        st.warning("No relation to draw with the current filters.")
    else:
        legend = " &nbsp; ".join(f"<span style='color:{c}'>●</span> {label}" for label, c in COLORS.items())
        st.markdown(f"{legend} &nbsp;|&nbsp; size = number of ALS articles &nbsp;|&nbsp; "
                    "dark border = drug tested in ALS &nbsp;|&nbsp; dashed = suggested",
                    unsafe_allow_html=True)
        st.iframe(draw_graph(view, profiles, center), height=640)

    if center:
        st.markdown(f"**{label_of[center]}**: {describe_als(profiles.get(center))}")
    shown = set(view["subject_id"]) | set(view["object_id"]) | ({center} if center else set())
    table = pd.concat([view, als_relations[als_relations["subject_id"].isin(shown)]])
    st.subheader(f"{len(table)} relations — select a row to see the evidence")
    show_evidence(table.sort_values("n_articles", ascending=False), key="explorer_table")

# ------------------------------------------------------------------ Statistics
else:
    st.title("Statistics")
    entity_types = pd.Series({**dict(zip(relations["subject_id"], relations["subject_type"])),
                              **dict(zip(relations["object_id"], relations["object_type"]))})
    counts = entity_types.value_counts()
    columns = st.columns(4)
    for column, label in zip(columns, ["Gene", "Drug", "Disease"]):
        column.metric(f"{label} nodes", int(counts.get(label, 0)))
    columns[3].metric("Relations (filtered)", len(filtered))

    left, right = st.columns(2)
    left.subheader("Relations by type")
    left.bar_chart(filtered["relation"].value_counts().reset_index(), x="relation", y="count",
                   horizontal=True, color=COLORS["Gene"], sort="-count")
    right.subheader("Outcome of ALS trials")
    trials = als_relations[als_relations["relation"] == "TESTED_FOR"]
    right.bar_chart(trials["outcome"].fillna("unknown").value_counts().reset_index(), x="outcome",
                    y="count", horizontal=True, color=COLORS["Drug"], sort="-count")

    left, right = st.columns(2)
    left.subheader("Top 20 genes associated with ALS")
    genes = als_relations[als_relations["relation"] == "ASSOCIATED_WITH"].nlargest(20, "n_articles")
    left.bar_chart(genes, x="subject", y="n_articles", horizontal=True, color=COLORS["Gene"],
                   sort="-n_articles")
    right.subheader("Drugs tested in ALS")
    right.dataframe(trials.nlargest(20, "n_articles")[["subject", "n_articles", "outcome"]]
                    .rename(columns={"subject": "drug", "n_articles": "articles"}),
                    hide_index=True, width="stretch")

    st.subheader("Drugs and their targets")
    st.dataframe(filtered[filtered["relation"] == "TARGETS"][["subject", "object", "n_articles", "certainty"]]
                 .rename(columns={"subject": "drug", "object": "target gene", "n_articles": "articles"})
                 .sort_values("articles", ascending=False), hide_index=True, width="stretch")

    with st.expander("Pipeline statistics (articles, rejected relations...)"):
        st.json(stats)
