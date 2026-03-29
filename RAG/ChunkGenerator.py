import sys
sys.path.insert(0, './')
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import json

def chunk_generator(files, chunk_size = 200,overlap = 40):
    chunks = []
    chapter_idx = ''
    for file_index, file in enumerate(files):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        chapters = re.split(r'\n(\d+(?:\.\d+){0,2})\s*\n', text)
        chapter_pairs = zip(chapters[1::2], chapters[2::2])
        chapter_idx_pattern = re.compile(r'\d+(?:\.\d+){0,2}')
        for chapter_idx, chapter in chapter_pairs:
            if chapter.count('\n') < 2:
                continue
            assert (chapter_idx_pattern.match(chapter_idx))

            words = chapter.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i: i + chunk_size]
                chunk_text = " ".join(chunk_words)
                chunk_id = "chunk_" + str(len(chunks)).zfill(5)
                chunk = {'id': chunk_id, "source": file, "chapter": chapter_idx, "start_word": i, "text": chunk_text}
                chunks.append(chunk)
    return chunks

def text_encode(chunks, model):
    texts_complete = ['passage: ' + chunk['text'] for chunk in chunks]
    # define the batch size to make (text) converting parallel
    embeddings = model.encode(texts_complete, batch_size = 32, normalize_embeddings = True)
    # currently it is okay to store chunks and embeddings in same jsonl file.
    # because the file size is still small, it won't take too much ram
    for chunk, emb in zip(chunks, embeddings):
        chunk['embedding'] = emb.tolist()
    return chunks

def chunk_to_jsonl(jsonfile, chunks):
    with open(jsonfile, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    doc_text_path = './chunks/'
    model = SentenceTransformer("intfloat/multilingual-e5-base", device = 'mps')
    file_List = [p.as_posix() for p in Path(doc_text_path).iterdir() if p.suffix == '.txt']
    chunks = chunk_generator(file_List, chunk_size = 200,overlap = 40)
    chunks = text_encode(chunks, model)
    chunk_to_jsonl(doc_text_path+'chunks.jsonl', chunks)

