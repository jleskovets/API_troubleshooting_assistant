import os
import chromadb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "troubleshooting_cases"


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def case_to_text(row):
    return f"""
API Area: {row['api_area']}
Endpoint: {row['endpoint']}
Error Code: {row['error_code']}
Problem: {row['problem']}
Root Cause: {row['root_cause']}
Solution: {row['solution']}
"""


def build_vector_store(csv_path="data/troubleshooting_cases_english.csv"):
    df = pd.read_csv(csv_path, sep=";")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():
        text = case_to_text(row)

        documents.append(text)
        embeddings.append(get_embedding(text))
        metadatas.append({
            "id": int(row["id"]),
            "api_area": str(row["api_area"]),
            "endpoint": str(row["endpoint"]),
            "error_code": str(row["error_code"]),
            "problem": str(row["problem"]),
            "root_cause": str(row["root_cause"]),
            "solution": str(row["solution"]),
        })
        ids.append(str(row["id"]))

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return len(ids)


def semantic_search(query, top_k=1):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    matches = []

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        score = round(1 - distance, 4)

        matches.append({
            "score": score,
            "case": metadata
        })

    return matches