"""Streamlit app to explore the mechanism-centered knowledge graph stored in Neo4j.

    streamlit run app.py
"""

import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network

load_dotenv(Path(__file__).resolve().parent / ".env")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

COLORS = {"Mechanism": "#7b4fd6", "Disease": "#1baf7a", "Molecule": "#eb6834"}
RELATION_TYPES = ["involved_in", "inhibits", "activates", "induces", "agonist_of",
                  "tested", "treatment", "documented_in_literature"]
MAX_EDGES = 200

st.set_page_config(page_title="Mechanism Knowledge Graph", page_icon="🧬", layout="wide")


@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


@st.cache_data
def load_data() -> pd.DataFrame:
    query = """
    MATCH (s:Entity)-[r]->(o:Entity)
    RETURN s.id AS subject_id, s.name AS subject, s.type AS subject_type,
           type(r) AS relation,
           o.id AS object_id, o.name AS object, o.type AS object_type,
           coalesce(r.source, 'llm') AS source,
           coalesce(r.n_articles, 0) AS n_articles,
           coalesce(r.certainty, 'established') AS certainty,
           r.outcome AS outcome, r.pmids AS pmids, r.evidences AS evidences,
           r.outcomes AS outcomes, r.years AS years
    """
    with get_driver().session() as session:
        rows = [record.data() for record in session.run(query)]
    columns = ["subject_id", "subject", "subject_type", "relation", "object_id", "object",
               "object_type", "source", "n_articles", "certainty", "outcome", "pmids",
               "evidences", "outcomes", "years"]
    data = pd.DataFrame(rows, columns=columns)
    for column in ("pmids", "evidences", "outcomes", "years"):
        data[column] = data[column].apply(lambda v: v if isinstance(v, list) else [])
    return data


def neighborhood(edges: pd.DataFrame, center: str, depth: int) -> pd.DataFrame:
    reached = {center}
    for _ in range(depth):
        touching = edges["subject_id"].isin(reached) | edges["object_id"].isin(reached)
        reached |= set(edges.loc[touching, "subject_id"]) | set(edges.loc[touching, "object_id"])
    return edges[edges["subject_id"].isin(reached) & edges["object_id"].isin(reached)]


def draw_graph(edges: pd.DataFrame, center: str | None) -> str:
    net = Network(height="620px", width="100%", directed=True, bgcolor="#ffffff",
                  font_color="#0b0b0b", cdn_resources="in_line")
    net.set_options("""{
      "nodes": {"scaling": {"label": {"drawThreshold": 1}}},
      "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}, "smooth": false},
      "physics": {"barnesHut": {"gravitationalConstant": -2500, "centralGravity": 0.6,
                                "springLength": 110, "avoidOverlap": 0.5},
                  "stabilization": {"iterations": 300}},
      "interaction": {"hover": true}
    }""")
    degree = pd.concat([edges["subject_id"], edges["object_id"]]).value_counts()
    for _, r in edges.iterrows():
        for node_id, name, label in ((r["subject_id"], r["subject"], r["subject_type"]),
                                     (r["object_id"], r["object"], r["object_type"])):
            mechanism = label == "Mechanism"
            net.add_node(node_id, label=name, color=COLORS.get(label, "#a3a29c"),
                         shape="box" if mechanism else "dot",
                         size=min(8 + 3 * math.sqrt(degree.get(node_id, 1)), 32),
                         title=f"{name} ({label}) — {degree.get(node_id, 0)} relations in view",
                         font={"size": 26 if node_id == center or mechanism else 18})
        manual = r["source"] == "manual"
        net.add_edge(r["subject_id"], r["object_id"],
                     title=f"{r['relation']} | " + ("manual" if manual else
                                                    f"{r['n_articles']} article(s) | {r['certainty']}"),
                     width=2 if manual else 1 + 2 * math.sqrt(r["n_articles"]),
                     color="#7b4fd6" if manual else "#a3a29c",
                     dashes=manual or r["certainty"] == "suggested")
    html = net.generate_html()
    # Freeze the layout once it is stable (otherwise the graph keeps moving)
    return html.replace(
        "network = new vis.Network(container, data, options);",
        "network = new vis.Network(container, data, options);\n"
        "network.once('stabilizationIterationsDone', function () {"
        " network.setOptions({physics: false}); });")


def show_evidence(table: pd.DataFrame, key: str) -> None:
    columns = ["subject", "relation", "object", "source", "n_articles", "certainty", "outcome"]
    selection = st.dataframe(table[columns], hide_index=True, width="stretch", key=key,
                             on_select="rerun", selection_mode="single-row")
    if selection.selection.rows:
        row = table.iloc[selection.selection.rows[0]]
        st.markdown(f"#### {row['subject']} → **{row['relation']}** → {row['object']}")
        if row["source"] == "manual":
            st.info("Edge added by hand (ALS is excluded from the PubMed query).")
        for pmid, evidence, year, outcome in zip(row["pmids"], row["evidences"], row["years"], row["outcomes"]):
            extra = f" — outcome: **{outcome}**" if outcome else ""
            st.markdown(f"- [PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) ({year}){extra}\n\n  > {evidence}")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🧬 Mechanism Knowledge Graph")
page = st.sidebar.radio("Page", ["Explorer", "Statistics"])

relations = load_data()
if relations.empty:
    st.info("Neo4j is empty. Run the pipeline, then `python neo4j/query_connection.py`.")
    st.stop()

st.sidebar.header("Filters")
types = st.sidebar.multiselect("Relation types", RELATION_TYPES, default=RELATION_TYPES)
certainties = st.sidebar.multiselect("Certainty", ["established", "suggested"],
                                     default=["established", "suggested"])
min_articles = st.sidebar.slider("Minimum number of articles", 1,
                                 max(int(relations["n_articles"].max()), 2), 1)
years = st.sidebar.slider("Publication years", 2015, 2026, (2020, 2026))

manual = relations["source"] == "manual"  # ALS edges: no article, never filtered out
in_years = relations["years"].apply(lambda ys: any(years[0] <= int(y) <= years[1] for y in ys if y))
filtered = relations[
    relations["relation"].isin(types)
    & (manual | (relations["certainty"].isin(certainties)
                 & (relations["n_articles"] >= min_articles) & in_years))
]

label_of = {}
for side in ("subject", "object"):
    label_of.update(zip(relations[f"{side}_id"], relations[side] + " (" + relations[f"{side}_type"] + ")"))

# ------------------------------------------------------------------ Explorer
if page == "Explorer":
    st.title("Explorer")
    st.caption("Mechanisms are boxes. ALS was added by hand at the end (purple dashed edges).")

    in_network = sorted(set(filtered["subject_id"]) | set(filtered["object_id"]),
                        key=lambda i: label_of[i].lower())
    left, right = st.columns([3, 1])
    center = left.selectbox("Search a mechanism, disease or molecule", [None] + in_network,
                            format_func=lambda i: "— whole network —" if i is None else label_of[i])
    depth = right.radio("Neighbors", [1, 2], horizontal=True, disabled=center is None,
                        help="2 = also the neighbors of the neighbors (e.g. molecule → mechanism → ALS)")

    if center:
        view = neighborhood(filtered, center, depth)
    else:
        view = pd.concat([filtered[filtered["source"] == "manual"],
                          filtered[filtered["source"] != "manual"].nlargest(MAX_EDGES, "n_articles")])
    if view.empty:
        st.warning("No relation to draw with the current filters.")
    else:
        legend = " &nbsp; ".join(f"<span style='color:{c}'>●</span> {label}" for label, c in COLORS.items())
        st.markdown(f"{legend} &nbsp;|&nbsp; edge width = number of articles &nbsp;|&nbsp; "
                    "dashed = suggested or manual", unsafe_allow_html=True)
        st.iframe(draw_graph(view, center), height=640)

    st.subheader(f"{len(view)} relations — select a row to see the evidence")
    show_evidence(view.sort_values("n_articles", ascending=False), key="explorer_table")

# ------------------------------------------------------------------ Statistics
else:
    st.title("Statistics")
    node_types = pd.Series({**dict(zip(relations["subject_id"], relations["subject_type"])),
                            **dict(zip(relations["object_id"], relations["object_type"]))})
    counts = node_types.value_counts()
    columns = st.columns(4)
    for column, label in zip(columns, ["Mechanism", "Disease", "Molecule"]):
        column.metric(f"{label} nodes", int(counts.get(label, 0)))
    columns[3].metric("Relations (filtered)", len(filtered))

    left, right = st.columns(2)
    left.subheader("Relations by type")
    left.bar_chart(filtered["relation"].value_counts().reset_index(), x="relation", y="count",
                   horizontal=True, color=COLORS["Mechanism"], sort="-count")
    right.subheader("Diseases per mechanism (involved_in)")
    involved = filtered[filtered["relation"] == "involved_in"]
    right.bar_chart(involved["subject"].value_counts().reset_index(), x="subject", y="count",
                    horizontal=True, color=COLORS["Disease"], sort="-count")

    st.subheader("Molecules and the mechanisms they act on")
    acting = filtered[filtered["object_type"] == "Mechanism"]
    if not acting.empty:
        table = acting.assign(effect=acting["relation"] + " (" + acting["n_articles"].astype(str) + ")")
        st.dataframe(table.pivot_table(index="subject", columns="object", values="effect",
                                       aggfunc=lambda v: ", ".join(v)).fillna(""),
                     width="stretch")

    st.subheader("Molecules tested or used in diseases")
    used = filtered[filtered["relation"].isin(["tested", "treatment"])]
    st.dataframe(used[["subject", "relation", "object", "outcome", "n_articles"]]
                 .rename(columns={"subject": "molecule", "object": "disease", "n_articles": "articles"})
                 .sort_values("articles", ascending=False), hide_index=True, width="stretch")
