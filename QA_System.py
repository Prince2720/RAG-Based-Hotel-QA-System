import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model_llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


with open("hotels_dataset.json") as f:
    data = json.load(f)

documents = []

for hotel in data:
    text = f"""
    Hotel: {hotel['hotel_name']}
    Location: {hotel['location']}
    Amenities: {', '.join(hotel['amenities'])}
    Rating: {hotel['rating']}
    Review: {hotel['review']}
    Policy: {hotel['policy']}
    """
    documents.append(text)

#print(documents)
print(documents[0])

for doc in documents:
    print(doc)


import re

def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks

all_chunks = []

for doc in documents:
    chunks = chunk_text(doc)
    all_chunks.extend(chunks)

# Clean the chunks to remove extra whitespace
cleaned_all_chunks = []
for chunk in all_chunks:
    cleaned_chunk = re.sub(r'\s+', ' ', chunk).strip()
    cleaned_all_chunks.append(cleaned_chunk)

def print_chunks(chunks):
    for i, chunk in enumerate(chunks):
        print(f"\n Chunk {i+1}")
        print("=" * 50)


        lines = chunk.split("\n")
        if len(lines) > 1:
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    print(f"{key.strip():<12} : {value.strip()}")
                else:
                    print(line.strip())
        else:
            print(chunk)

        print("=" * 50)

print_chunks(cleaned_all_chunks)



from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = cleaned_all_chunks  # your chunks

embeddings = model.encode(texts)

print("Shape:", embeddings.shape)   # (num_chunks, 384)
#print("\nFirst embedding:\n", embeddings[0])

for i, emb in enumerate(embeddings):
    print(f"\n{'='*50}")
    print(f"Chunk {i+1}")
    print(all_chunks[i])

    print("\nEmbedding preview:")
    print([round(x, 3) for x in emb[:10]])


import faiss
import numpy as np

# embeddings from previous step
# embeddings.shape = (num_chunks, 384)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)   # L2 distance

index.add(np.array(embeddings))        # store embeddings

print("Total vectors in index:", index.ntotal)


def retrieve(query, k=3):
    # convert query → embedding
    query_embedding = model.encode([query])

    # search in FAISS
    distances, indices = index.search(np.array(query_embedding), k)

    # get matching chunks
    results = [cleaned_all_chunks[i] for i in indices[0]]

    return results

#user_query = input("Enter your query: ")
#results = retrieve(user_query, k=3)

#print("\nTop 3 matching chunks for query:", user_query)
#for i, result in enumerate(results):
    print(f"\nResult {i+1}:\n{result}")

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks[:2])

    prompt = f"""
Answer the question based ONLY on the context.

Context:
{context}

Question: {query}

Give a short and clear answer.
"""
    return prompt


def generate_answer(query):
    retrieved_chunks = retrieve(query, k=3)

    prompt = build_prompt(query, retrieved_chunks)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model_llm.generate(**inputs, max_new_tokens=100)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if answer == "":
        answer = "Not enough information available."

    return answer, retrieved_chunks

#  EVALUATION

test_queries = [
    {
        "query": "hotel with free wifi and breakfast",
        "expected_keywords": ["wifi", "breakfast"]
    },
    {
        "query": "hotel near beach with good reviews",
        "expected_keywords": ["beach", "rating"]
    },
    {
        "query": "cancellation policy of hotel",
        "expected_keywords": ["policy", "cancellation"]
    }
]


def is_relevant(chunk, keywords):
    chunk_lower = chunk.lower()
    return any(keyword in chunk_lower for keyword in keywords)


def precision_at_k(query, expected_keywords, k=3):
    retrieved_chunks = retrieve(query, k)

    relevant_count = 0
    for chunk in retrieved_chunks:
        if is_relevant(chunk, expected_keywords):
            relevant_count += 1

    precision = relevant_count / k
    return precision, retrieved_chunks



mode = input("Choose mode: (1) Query  (2) Evaluate : ")

if mode == "1":
    # Interactive Query Mode
    while True:
        user_query = input("\nEnter your query (or type 'exit'): ")

        if user_query.lower() == "exit":
            break

        answer, context = generate_answer(user_query)

        print("\n Retrieved Context:")
        for i, c in enumerate(context):
            print(f"\nChunk {i+1}:")
            print(c)

        print("\n Final Answer:")
        print(answer)


elif mode == "2":
    #  Evaluation Mode
    for test in test_queries:
        query = test["query"]
        keywords = test["expected_keywords"]

        precision, chunks = precision_at_k(query, keywords, k=3)
        answer, _ = generate_answer(query)

        print("\n==============================")
        print("Query:", query)
        print("Precision@3:", round(precision, 2))

        print("\n Answer:", answer)

        print("\n Retrieved Chunks:")
        for i, c in enumerate(chunks):
            print(f"\nChunk {i+1}: {c[:100]}...")