from sentence_transformers import SentenceTransformer
import chromadb
import ollama

def similarity_search(query, model, collection, k=5):
    query = "query: " +  query
    query_embedding = model.encode(query, normalize_embeddings = True)
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = k
    )
    return results

def load_db(vector_db_dir = './vector_db', collection_name = 'rag_collection'):
    client = chromadb.PersistentClient(path = vector_db_dir)
    collection = client.get_collection(name = collection_name)

    return collection

def build_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(retrieved_chunks)
    # be aware the prompt text is not just text, it is also an instruction to tell the llm how to generation answeer
    # few hallucination
    prompt = f"""
    Du bist ein hilfreicher Assistent.

    Answer the question based only on the context below. Include all formulas and steps exactly as in the context.
    Wenn die Antwort nicht im Kontext enthalten ist, sage genau: "Ich weiß es nicht."
    
    Kontext:
    ---
    {context}
    ---
    Frage: {query}
    
    Antwort:
    """
    return prompt

def ask_ollama(prompt):
    response = ollama.chat(
        model = "llama3",
        messages = [{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']
    
    
if __name__ == '__main__':
    model = SentenceTransformer("intfloat/multilingual-e5-base", device='mps')
    collection = load_db()
    #query = "Was ist die Definition von Schwimmwinkel?"
    #query = "wie kann ich den Schwimmwinkel schätzen"
    #query = "was ist einspurmodell"
    #query = "wenn ist das fahrzeug übersteuert"
    #query = "nenn mir Fälle, wenn das Fahrzeug überstuert ist"
    #query = "wie können die reibwerte bestimmt werden"
    query = "erklär mir die UKF"
    retrieved_chunks = similarity_search(query, model, collection)
    prompt = build_prompt(query, retrieved_chunks['documents'][0])
    ollama_answer = ask_ollama(prompt)
    print(ollama_answer)