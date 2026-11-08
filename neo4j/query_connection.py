"""Load data/association.json into Neo4j.

    python neo4j/query_connection.py           # adds to the existing graph
    python neo4j/query_connection.py --reset   # empties the database first
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
ASSOCIATION_FILE = ROOT / "data" / "association.json"
load_dotenv(ROOT / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

NODE_TYPES = {"Mechanism", "Disease", "Molecule"}


def edge_properties(item: dict) -> dict:
    """Neo4j does not accept null inside lists."""
    return {
        "source": item.get("source", "llm"),
        "certainty": item.get("certainty", "unknown"),
        "n_articles": item.get("n_articles", 0),
        "outcome": item.get("outcome"),
        "pmids": [str(p) for p in item.get("pmids", []) if p is not None],
        "evidences": [e or "" for e in item.get("evidences", [])],
        "outcomes": [o or "" for o in item.get("outcomes", [])],
        "years": [int(y) if y else 0 for y in item.get("years", [])],
    }


def load(path: Path, reset: bool) -> None:
    if not path.exists():
        raise SystemExit(f"{path} not found: run src/associations_llm.py first.")
    associations = [a for a in json.loads(path.read_text("utf-8"))["associations"] if not a["negated"]]

    # Labels and relation types cannot be query parameters: one query per (relation, types)
    groups = defaultdict(list)
    for a in associations:
        key = (a["relation"], a["subject_type"], a["object_type"])
        if not re.fullmatch(r"[a-z_]+", a["relation"]) or not {key[1], key[2]} <= NODE_TYPES:
            print(f"Skipped (unexpected relation or type): {key}")
            continue
        groups[key].append({"subject_id": a["subject_id"], "subject": a["subject"],
                            "object_id": a["object_id"], "object": a["object"],
                            "props": edge_properties(a)})

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            if reset:
                session.run("MATCH (n) DETACH DELETE n")
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
            for (relation, subject_type, object_type), rows in groups.items():
                session.run(f"""
                    UNWIND $rows AS row
                    MERGE (s:Entity {{id: row.subject_id}})
                    SET s:{subject_type}, s.name = row.subject, s.type = '{subject_type}'
                    MERGE (o:Entity {{id: row.object_id}})
                    SET o:{object_type}, o.name = row.object, o.type = '{object_type}'
                    MERGE (s)-[r:{relation}]->(o)
                    SET r += row.props
                """, rows=rows)
                print(f"{len(rows):>5}  {subject_type} -[{relation}]-> {object_type}")
    finally:
        driver.close()
    print(f"Done: {sum(len(r) for r in groups.values())} relations loaded into Neo4j.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=ASSOCIATION_FILE)
    parser.add_argument("--reset", action="store_true", help="Delete everything in Neo4j before loading")
    args = parser.parse_args()
    load(args.file, args.reset)
