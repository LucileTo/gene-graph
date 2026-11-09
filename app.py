"""Streamlit app to explore the mechanism-centered knowledge graph stored in Neo4j.

    streamlit run app.py
"""

import json
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


# Injected in the pyvis page: black edges on click, Ctrl+click = recenter, reset button.
# __ALS__ is replaced by the id of the ALS node (or null).
INTERACTION_JS = """
(function () {
  var ALS = __ALS__;
  var BLACK = {color: '#000000', highlight: '#000000', hover: '#000000'};
  var RING_GAP = 170;   // minimum distance between two rings
  var ARC = 100;        // minimum distance between two nodes of the same ring

  // Original style of every node and edge, to restore it later
  var nodeStyle = {}, edgeStyle = {};
  nodes.get().forEach(function (n) {
    nodeStyle[n.id] = {id: n.id, color: n.color, size: n.size, borderWidth: n.borderWidth, font: n.font};
  });
  edges.get().forEach(function (e) {
    edgeStyle[e.id] = {id: e.id, color: e.color, width: e.width, dashes: e.dashes};
  });

  function restoreStyles() {
    nodes.update(Object.values(nodeStyle));
    edges.update(Object.values(edgeStyle));
  }

  function highlight(id) {
    restoreStyles();
    var connected = network.getConnectedNodes(id);
    edges.update(network.getConnectedEdges(id).map(function (e) {
      return {id: e, color: BLACK, width: edgeStyle[e].width + 2};
    }));
    nodes.update(connected.concat([id]).map(function (n) {
      var font = nodeStyle[n].font || {};
      return {id: n, borderWidth: 4,
              color: {background: nodeStyle[n].color, border: '#000000'},
              font: {size: (font.size || 18) + 4, color: font.color || '#0b0b0b',
                     strokeWidth: 1, strokeColor: font.color || '#0b0b0b'}};
    }));
  }

  // ---- radial layout: the centre in the middle, then one ring per distance (1 link, 2 links...)
  function radialLayout(centerId) {
    var ids = nodes.getIds(), level = {}, queue = [centerId], target = {};
    level[centerId] = 0;
    while (queue.length) {                       // breadth-first search from the centre
      var current = queue.shift();
      network.getConnectedNodes(current).forEach(function (n) {
        if (level[n] === undefined) { level[n] = level[current] + 1; queue.push(n); }
      });
    }
    var maxLevel = Math.max.apply(null, Object.values(level));
    ids.forEach(function (id) { if (level[id] === undefined) level[id] = maxLevel + 1; });  // not linked

    target[centerId] = {x: 0, y: 0};
    var angle = {}, radius = 0;
    angle[centerId] = 0;
    for (var k = 1; k <= Math.max.apply(null, Object.values(level)); k++) {
      var ring = ids.filter(function (id) { return level[id] === k; });
      if (!ring.length) continue;
      // wanted angle = mean angle of the neighbours on the inner ring (children near their parent)
      var wanted = {};
      ring.forEach(function (id) {
        var parents = network.getConnectedNodes(id).filter(function (n) { return level[n] === k - 1; });
        var x = 0, y = 0;
        parents.forEach(function (n) { x += Math.cos(angle[n]); y += Math.sin(angle[n]); });
        wanted[id] = parents.length && (x || y) ? Math.atan2(y, x) : 0;
      });
      ring.sort(function (a, b) { return wanted[a] - wanted[b]; });
      radius = Math.max(radius + RING_GAP, ring.length * ARC / (2 * Math.PI));
      var offset = k === 1 ? -Math.PI / 2 : wanted[ring[0]];
      ring.forEach(function (id, i) {
        angle[id] = offset + 2 * Math.PI * i / ring.length;
        target[id] = {x: radius * Math.cos(angle[id]), y: radius * Math.sin(angle[id])};
      });
    }
    return target;
  }

  // Move every node to its target position, smoothly, then fit the view
  function moveAll(target, animate) {
    var ids = Object.keys(target), start = {};
    ids.forEach(function (id) { start[id] = network.getPosition(id); });
    var t0 = null, duration = animate ? 800 : 0;
    function step(now) {
      if (t0 === null) t0 = now;                 // same clock as requestAnimationFrame
      var k = duration ? Math.min(1, Math.max(0, (now - t0) / duration)) : 1;
      k = k * (2 - k);                           // ease out
      ids.forEach(function (id) {
        network.moveNode(id, start[id].x + (target[id].x - start[id].x) * k,
                             start[id].y + (target[id].y - start[id].y) * k);
      });
      if (k < 1) requestAnimationFrame(step);
      else network.fit({animation: animate});
    }
    requestAnimationFrame(step);
  }

  function centerOn(id, animate) {
    network.setOptions({physics: false});        // the layout is computed, no simulation
    moveAll(radialLayout(id), animate);
  }

  window.kgReset = function () {
    restoreStyles();
    network.unselectAll();
    if (ALS !== null) centerOn(ALS, true); else network.fit({animation: true});
  };

  network.on('click', function (params) {
    if (!params.nodes.length) { restoreStyles(); return; }
    var id = params.nodes[0];
    highlight(id);
    if (params.event && params.event.srcEvent && params.event.srcEvent.ctrlKey) centerOn(id, true);
  });

  if (ALS !== null) centerOn(ALS, false);        // ALS in the middle from the start
})();
"""

RESET_BUTTON = (
    '<button onclick="kgReset()" style="position:absolute;top:8px;left:8px;z-index:10;'
    'padding:6px 12px;border:1px solid #a3a29c;border-radius:6px;background:#fff;'
    'cursor:pointer;font:14px sans-serif">↺ Reset graph</button>'
)


def draw_graph(edges: pd.DataFrame, center: str | None) -> str:
    # No font_color here: pyvis would overwrite the per-node font (size) with it
    net = Network(height="620px", width="100%", directed=True, bgcolor="#ffffff",
                  cdn_resources="in_line")
    net.set_options("""{
      "nodes": {"font": {"color": "#0b0b0b"}, "scaling": {"label": {"drawThreshold": 1}}},
      "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}, "smooth": false},
      "physics": {"barnesHut": {"gravitationalConstant": -2500, "centralGravity": 0.6,
                                "springLength": 110, "avoidOverlap": 0.5},
                  "stabilization": {"iterations": 300}},
      "interaction": {"hover": true}
    }""")
    degree = pd.concat([edges["subject_id"], edges["object_id"]]).value_counts()
    # ALS is only a landmark: the object of the manual edges, drawn big and fixed in the middle
    manual_edges = edges[edges["source"] == "manual"]
    als_id = manual_edges["object_id"].iloc[0] if not manual_edges.empty else None

    for _, r in edges.iterrows():
        for node_id, name, label in ((r["subject_id"], r["subject"], r["subject_type"]),
                                     (r["object_id"], r["object"], r["object_type"])):
            options = dict(
                label=name, color=COLORS.get(label, "#a3a29c"), borderWidth=1,
                shape="box" if label == "Mechanism" else "dot",
                size=min(8 + 3 * math.sqrt(degree.get(node_id, 1)), 32),
                title=f"{name} ({label}) — {degree.get(node_id, 0)} relations in view",
                font={"size": 26 if node_id == center or label == "Mechanism" else 18})
            if node_id == als_id:
                options.update(shape="circle", label="SLA", color="#d62728", margin=22,
                               font={"size": 34, "color": "#ffffff"},
                               title="SLA / ALS (Amyotrophic Lateral Sclerosis), added by hand")
            net.add_node(node_id, **options)
        manual = r["source"] == "manual"
        net.add_edge(r["subject_id"], r["object_id"],
                     title=f"{r['relation']} | " + ("manual" if manual else
                                                    f"{r['n_articles']} article(s) | {r['certainty']}"),
                     width=3 if manual else 1 + 2 * math.sqrt(r["n_articles"]),
                     color="#d62728" if manual else "#a3a29c",
                     dashes=bool(manual or r["certainty"] == "suggested"))
    html = net.generate_html()
    script = INTERACTION_JS.replace("__ALS__", json.dumps(als_id))
    # Freeze the layout once it is stable (otherwise the graph keeps moving), then add the interactions
    html = html.replace(
        "network = new vis.Network(container, data, options);",
        "network = new vis.Network(container, data, options);\n"
        "network.once('stabilizationIterationsDone', function () {"
        " network.setOptions({physics: false}); });\n" + script)
    return html.replace("</body>", RESET_BUTTON + "</body>")


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
    st.caption("Mechanisms are boxes. ALS was added by hand at the end (red dashed edges). **Click** a node: its edges turn black. **Ctrl+click**: put it in the middle, the nodes linked to it form the first circle. **Reset graph** (top left of the graph): ALS back in the middle.")

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
                    "dashed = suggested &nbsp;|&nbsp; <span style='color:#d62728'>red dashed</span> = ALS (manual)", unsafe_allow_html=True)
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
