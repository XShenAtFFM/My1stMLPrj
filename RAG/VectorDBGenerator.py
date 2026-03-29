import chromadb
import json
from chromadb.config import Settings

def build_vector_db(chunk_file, vector_db_dir = './vector_db'):
    client = chromadb.PersistentClient(path = vector_db_dir)

    collection = client.get_or_create_collection(
        name = 'rag_collection',
        metadata = {"hnsw:space": "cosine"}
    )

    ids = []
    documents = []
    embeddings = []

    with open(chunk_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            ids.append(data['id']),
            documents.append(data['text']),
            embeddings.append(data['embedding'])

    collection.add(
        ids = ids,
        documents = documents,
        embeddings = embeddings
    )

if __name__ == '__main__':
    build_vector_db('./chunks/chunks.jsonl', './vector_db')


