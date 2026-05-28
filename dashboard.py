import streamlit as st
import pandas as pd
import os
from pypdf import PdfReader
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config

# Import your core logic and LLM from main.py
from main import extract_and_save, llm

# --- DASHBOARD CONFIG ---
st.set_page_config(layout="wide", page_title="MediGraph Pro Dashboard", page_icon="🏥")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 MediGraph AI Assistant")
st.caption("Final Year Project | VIT Bhopal | Batch 2022-2026")
st.markdown("---")

# --- DATABASE CONNECTION HELPER ---
def get_neo4j_data():
    nodes, edges = [], []
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        with driver.session() as session:
            result = session.run("MATCH (n)-[r]->(m) RETURN n.name as src, type(r) as rel, m.name as tgt")
            seen_nodes = set()
            for record in result:
                for key in ['src', 'tgt']:
                    if record[key] not in seen_nodes:
                        color = "#0072B2" if key == 'src' else "#D55E00"
                        nodes.append(Node(id=record[key], label=record[key], size=20, color=color))
                        seen_nodes.add(record[key])
                edges.append(Edge(source=record['src'], target=record['tgt'], label=record['rel']))
        driver.close()
    except Exception as e:
        st.error(f"Neo4j Connection Error: {e}")
    return nodes, edges

# --- SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("📁 Data Ingestion")
    st.write("Upload clinical notes to expand the Knowledge Graph.")
    uploaded_files = st.file_uploader("Upload PDFs or CSVs", type=["pdf", "csv"], accept_multiple_files=True)
    
    if st.button("🚀 Process & Sync"):
        if not uploaded_files:
            st.warning("Please upload a file first.")
        else:
            for file in uploaded_files:
                with st.status(f"Analyzing {file.name}...", expanded=True) as status:
                    content = ""
                    if file.name.endswith(".pdf"):
                        reader = PdfReader(file)
                        content = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
                    else:
                        df = pd.read_csv(file)
                        content = df.to_string()
                    
                    extract_and_save(content)
                    status.update(label=f"✅ {file.name} Processed!", state="complete")
            st.rerun()

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("💬 Clinical Chat Agent")
    
    # Simple Chat interface
    chat_container = st.container(height=500)
    user_query = st.chat_input("Ask: 'What are the symptoms of the Flu?'")
    
    if user_query:
        chat_container.chat_message("user").write(user_query)
        with st.spinner("Consulting Knowledge Graph..."):
            # RAG Logic: Pull facts from Neo4j to give Gemini context
            try:
                driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
                with driver.session() as session:
                    res = session.run("MATCH (n)-[r]->(m) RETURN n.name + ' ' + type(r) + ' ' + m.name as fact")
                    context = ". ".join([r['fact'] for r in res])
                driver.close()
                
                # Prompt Engineering
                full_prompt = f"""
                You are a medical assistant. Use the following extracted knowledge graph facts to answer the question.
                If the answer isn't in the facts, say you don't know based on the current graph.
                
                FACTS: {context}
                QUESTION: {user_query}
                """
                ai_response = llm.invoke(full_prompt)
                chat_container.chat_message("assistant").write(ai_response.content)
            except Exception as e:
                chat_container.error(f"Chat Logic Error: {e}")

with col2:
    st.header("🕸️ Knowledge Graph")
    
    nodes, edges = get_neo4j_data()
    
    # Stats row
    s1, s2 = st.columns(2)
    s1.metric("Total Entities", len(nodes))
    s2.metric("Total Relations", len(edges))
    
    if nodes:
        config = Config(
            width=800, 
            height=600, 
            directed=True, 
            nodeHighlightBehavior=True, 
            collapsible=True,
            staticGraph=False
        )
        agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.info("The graph is currently empty. Upload data to see the visualization.")