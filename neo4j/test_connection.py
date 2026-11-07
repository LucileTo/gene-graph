import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load .env
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print(" Try Connecting To Neo4j...")
print(f"URI: {NEO4J_URI}")
print(f"User: {NEO4J_USER}")

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    driver.verify_connectivity()
    
    print("\n Connection Succerful! to Neo 4j")
    
    driver.close()
except Exception as e:
    print(f"\n Connection failed  error:\n{e}")