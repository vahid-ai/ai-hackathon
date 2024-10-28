from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import duckdb
import uuid
from datetime import datetime
import networkx as nx
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings
)
from llama_index.core.storage import StorageContext
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.indices.query.schema import QueryType
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from typing import List, Dict
import re
import os

# Configure Ollama
llm = Ollama(model="llama3.1", request_timeout=60.0)
embed_model = OllamaEmbedding(model_name="llama3.1")
Settings.llm = llm
Settings.embed_model = embed_model

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321"],  # Astro's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DuckDB
conn = duckdb.connect('form_submissions.db')

# Create table if it doesn't exist
conn.execute('''
    CREATE TABLE IF NOT EXISTS submissions (
        id VARCHAR PRIMARY KEY,
        name VARCHAR,
        organization VARCHAR,
        interests VARCHAR,
        bio VARCHAR,
        submission_date TIMESTAMP
    )
''')

# Initialize graph store and knowledge graph
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)
# Correctly configure settings
service_context = Settings
service_context.llm = llm
service_context.embed_model = embed_model
kg_index = None

def extract_entities_and_relations(text: str, context: Dict) -> List[Dict]:
    """Extract entities and their relationships from text."""
    # Use LLM to extract structured information
    prompt = f"""
    Extract key information and relationships from the following text. 
    Format the response as a list of relationship triplets (subject, predicate, object).
    
    Text: {text}
    Context: Person named {context.get('name')} from organization {context.get('organization')}
    
    Example output format:
    - (Person, works_at, Organization)
    - (Person, interested_in, Topic)
    - (Person, has_expertise, Skill)
    """
    
    try:
        response = llm.complete(prompt)
        # Parse the response to extract triplets
        triplets = []
        for line in response.text.split('\n'):
            if '(' in line and ')' in line:
                # Extract content between parentheses
                triplet_str = line[line.find('(')+1:line.find(')')].strip()
                if ',' in triplet_str:
                    subj, pred, obj = [x.strip() for x in triplet_str.split(',', 2)]
                    triplets.append({
                        "subject": subj,
                        "predicate": pred,
                        "object": obj
                    })
        return triplets
    except Exception as e:
        print(f"Error extracting relationships: {str(e)}")
        return []

def create_knowledge_graph_from_submission(submission: Dict) -> List[Dict]:
    """Create knowledge graph relationships from a submission."""
    relationships = []
    
    # Basic relationships
    relationships.append({
        "subject": submission["name"],
        "predicate": "works_at",
        "object": submission["organization"]
    })
    
    # Extract interests
    interests = [interest.strip() for interest in submission["interests"].split(',')]
    for interest in interests:
        relationships.append({
            "subject": submission["name"],
            "predicate": "interested_in",
            "object": interest
        })
    
    # Extract additional relationships from bio
    bio_relationships = extract_entities_and_relations(
        submission["bio"],
        {"name": submission["name"], "organization": submission["organization"]}
    )
    relationships.extend(bio_relationships)
    
    return relationships

def initialize_knowledge_graph():
    global kg_index
    # Create knowledge graph from submissions
    result = conn.execute('''
        SELECT * FROM submissions
    ''').fetchall()
    
    # Convert submissions to documents and extract relationships
    documents = []
    all_relationships = []
    
    for row in result:
        submission = {
            "id": row[0],
            "name": row[1],
            "organization": row[2],
            "interests": row[3],
            "bio": row[4],
            "submission_date": row[5]
        }
        
        # Create document
        text = f"Name: {row[1]}\nOrganization: {row[2]}\nInterests: {row[3]}\nBio: {row[4]}"
        metadata = {
            "name": row[1],
            "organization": row[2],
            "submission_date": str(row[5]) if row[5] else None
        }
        doc = Document(text=text, id_=row[0], metadata=metadata)
        documents.append(doc)
        
        # Extract relationships
        relationships = create_knowledge_graph_from_submission(submission)
        all_relationships.extend(relationships)
    
    if documents:
        # Create knowledge graph index
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            service_context=service_context,
            max_triplets_per_chunk=10,
            include_embeddings=True
        )
        
        # Add extracted relationships to the graph
        for rel in all_relationships:
            try:
                kg_index.add_triplet(rel["subject"], rel["predicate"], rel["object"])
            except Exception as e:
                print(f"Error adding relationship: {str(e)}")
    
    return kg_index

class FormData(BaseModel):
    name: str
    organization: str
    interests: str
    bio: str

@app.post("/submit")
async def submit_form(data: FormData):
    # Generate unique ID
    submission_id = str(uuid.uuid4())
    
    # Get current timestamp
    submission_date = datetime.now()
    
    # Insert data into DuckDB
    conn.execute('''
        INSERT INTO submissions (id, name, organization, interests, bio, submission_date)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', [submission_id, data.name, data.organization, data.interests, data.bio, submission_date])
    
    # Extract relationships and update knowledge graph
    submission = {
        "id": submission_id,
        "name": data.name,
        "organization": data.organization,
        "interests": data.interests,
        "bio": data.bio,
        "submission_date": submission_date
    }
    
    # Create document for the new submission
    text = f"Name: {data.name}\nOrganization: {data.organization}\nInterests: {data.interests}\nBio: {data.bio}"
    doc = Document(text=text, id_=submission_id)
    
    # Initialize knowledge graph if not exists
    if kg_index is None:
        initialize_knowledge_graph()
    else:
        # Add new document to existing knowledge graph
        kg_index.insert(doc)
        
        # Extract and add new relationships
        relationships = create_knowledge_graph_from_submission(submission)
        for rel in relationships:
            try:
                kg_index.add_triplet(rel["subject"], rel["predicate"], rel["object"])
            except Exception as e:
                print(f"Error adding relationship: {str(e)}")
    
    return {
        "status": "success",
        "message": f"Thank you {data.name}, your submission has been recorded!",
        "submission_id": submission_id
    }

@app.get("/submissions")
async def get_submissions():
    # Fetch all submissions
    result = conn.execute('''
        SELECT * FROM submissions
        ORDER BY submission_date DESC
    ''').fetchall()
    
    # Convert to list of dictionaries
    submissions = []
    for row in result:
        submissions.append({
            "id": row[0],
            "name": row[1],
            "organization": row[2],
            "interests": row[3],
            "bio": row[4],
            "submission_date": row[5].isoformat() if row[5] else None
        })
    
    return submissions

@app.get("/query")
async def query_knowledge_graph(query: str):
    if kg_index is None:
        initialize_knowledge_graph()
    
    if kg_index is None:
        return {"error": "No data available for querying"}
    
    try:
        # Create query engine with hybrid search
        query_engine = kg_index.as_query_engine(
            response_mode="tree",
            verbose=True,
            similarity_top_k=3,  # Number of similar nodes to consider
            include_text=True    # Include raw text in response
        )
        
        # Execute query
        response = query_engine.query(query)
        
        # Extract source nodes and metadata
        source_nodes = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                node_info = {
                    "node_id": node.node_id,
                    "score": node.score if hasattr(node, 'score') else None,
                    "text": node.text,
                }
                if hasattr(node, 'metadata'):
                    node_info.update(node.metadata)
                source_nodes.append(node_info)
        
        # Get related relationships from the graph
        graph_response = kg_index.get_relationships(response.response, max_hops=2)
        
        return {
            "response": str(response),
            "source_nodes": source_nodes,
            "relationships": graph_response if graph_response else []
        }
    except Exception as e:
        return {"error": str(e)}

@app.on_event("shutdown")
async def shutdown_event():
    # Close DuckDB connection
    conn.close()
    # Close Neo4j connection
    if graph_store:
        graph_store.close()

if __name__ == "__main__":
    import uvicorn
    # Initialize knowledge graph on startup
    initialize_knowledge_graph()
    uvicorn.run(app, host="0.0.0.0", port=8000)
