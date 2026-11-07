import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class KnowledgeGraphLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_associations(self, file_path):
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} tidak ditemukan!")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Mengambil list dari key 'associations'
        associations = raw_data.get("associations", [])
        print(f" Memproses {len(associations)} asosiasi dari {file_path}...")

       
        with self.driver.session() as session:
            for item in associations:
                sub_id = item["subject_id"]
                sub_name = item["subject"]
                sub_type = item["subject_type"]
                
                obj_id = item["object_id"]
                obj_name = item["object"]
                obj_type = item["object_type"]
                
                relation_type = item["relation"]
                certainty = item.get("certainty", "unknown")
                n_articles = item.get("n_articles", 1)
                negated = item.get("negated", False)
                outcome = item.get("outcome", None)
                
                # Pembersihan data agar bebas dari null (None)
                pmids = [str(p) if p is not None else "" for p in item.get("pmids", [])]
                evidences = [str(e) if e is not None else "" for e in item.get("evidences", [])]
                outcomes = [str(o) if o is not None else "" for o in item.get("outcomes", [])]
                years = [int(y) if y is not None else 0 for y in item.get("years", [])]
                
                query = f"""
                MERGE (s:Entity {{id: $sub_id}})
                ON CREATE SET s.name = $sub_name, s.type = $sub_type
                
                MERGE (o:Entity {{id: $obj_id}})
                ON CREATE SET o.name = $obj_name, o.type = $obj_type
                
                MERGE (s)-[r:{relation_type}]->(o)
                SET r.certainty = $certainty, 
                      r.n_articles = $n_articles,
                      r.negated = $negated,
                      r.outcome = $outcome,
                      r.pmids = $pmids,
                      r.evidences = $evidences,
                      r.outcomes = $outcomes,
                      r.years = $years
                """
                
                session.run(
                    query,
                    sub_id=sub_id, sub_name=sub_name, sub_type=sub_type,
                    obj_id=obj_id, obj_name=obj_name, obj_type=obj_type,
                    certainty=certainty, n_articles=n_articles,
                    negated=negated, outcome=outcome,
                    pmids=pmids, evidences=evidences, outcomes=outcomes, years=years
                )
        print(" Sukses! Semua asosiasi berhasil dimasukkan ke Neo4j.")

if __name__ == "__main__":
    loader = KnowledgeGraphLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        loader.import_associations("C:/Users/herur/Documents/UBE/ALS_TRANSVERSAL/ROOT/gene-graph/data/association.json")
    finally:
        loader.close()