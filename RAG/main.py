from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from SimilaritySearch import similarity_search, load_db, ask_ollama, build_prompt
from sentence_transformers import SentenceTransformer
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

try:
    model = SentenceTransformer("intfloat/multilingual-e5-base", device='mps')
    collection = load_db()
except Exception as e:
    print("Startup error:", e)
    raise

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query")
async def get_query(query_request: QueryRequest):
    user_query = query_request.query
    ollama_answer = await run_in_threadpool(answer_generation, user_query)
    #ollama_answer = "test"
    return {"answer": ollama_answer}


def answer_generation(query):
    retrieved_chunks = similarity_search(query, model, collection)
    findings = retrieved_chunks.get("documents", [])
    if not findings or not findings[0]:
        return "No relevant information found"
    prompt = build_prompt(query, findings[0])
    return ask_ollama(prompt)
