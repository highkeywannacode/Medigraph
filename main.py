import os
from typing import List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Config
URI = "bolt://127.0.0.1:7687" # Using direct IP to avoid IPv6 issues
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
KEY = os.getenv("GOOGLE_API_KEY")

class MedicalRelationship(BaseModel):
    source: str = Field(description="The starting entity")
    target: str = Field(description="The connected entity")
    relation: str = Field(description="Type: 'HAS_SYMPTOM', 'TREATS', 'CAUSES'")

class MedicalGraph(BaseModel):
    relationships: List[MedicalRelationship]

# Use the version that works for you!
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=KEY)

def extract_and_save(text):
    structured_llm = llm.with_structured_output(MedicalGraph)
    print(f"🤖 Reading clinical note...")
    result = structured_llm.invoke(f"Extract medical relationships: {text}")
    
    # Connect using the direct IP
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    with driver.session() as session:
        for rel in result.relationships:
            query = (
                "MERGE (a:Medical {name: $source}) "
                "MERGE (b:Medical {name: $target}) "
                f"MERGE (a)-[:{rel.relation}]->(b)"
            )
            session.run(query, source=rel.source, target=rel.target)
            print(f"✅ Saved: {rel.source} -> {rel.target}")
    driver.close()

if __name__ == "__main__":
    note = "Patient has fever. Diagnosed with Flu."
    try:
        extract_and_save(note)
        print("🚀 Success! Check Neo4j Browser now.")
    except Exception as e:
        print(f"❌ Still failing: {e}")