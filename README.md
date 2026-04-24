**🏨 RAG-Based Hotel Q&A System**

**📌Overview**

This project implements a Retrieval-Augmented Generation (RAG) system that answers user queries about hotels using a custom dataset.
The system retrieves relevant hotel information using vector similarity search and generates accurate, context-based answers using a language model.

**🎯 Objective**
- Build a question-answering system over hotel data.
- Ensure answers are grounded in retrieved context.
- Reduce hallucination using prompt constraints.

**📂 Dataset**

A synthetic dataset of 50 hotel records containing:
- Hotel name
- Location
- Amenities
- Ratings
- Reviews
- Policies

**⚙️ Tech Stack**

- Python
- Sentence Transformers (all-MiniLM-L6-v2) → embeddings
- FAISS → vector database
- Hugging Face Transformers (FLAN-T5) → answer generation
- NumPy, JSON

**🧠 System Architecture**

User Query
   ↓
Embedding (SentenceTransformer)
   ↓
FAISS Vector Search
   ↓
Top-K Relevant Chunks
   ↓
Prompt Construction
   ↓
FLAN-T5 (LLM)
   ↓
Final Answer

**🔧 Implementation Steps**

1️⃣ Preprocessing
- Converted dataset into structured text documents
- Applied text chunking (200 chars, 50 overlap)

👉 Reason:
- Maintains context continuity
- Improves retrieval accuracy

2️⃣ Embedding & Storage
- Generated embeddings using all-MiniLM-L6-v2
- Stored vectors in FAISS IndexFlatL2
  
3️⃣ Retrieval
- Converted user query into embedding
- Retrieved Top-3 most similar chunks
  
4️⃣ Generative QA

- Used FLAN-T5 model
- Prompt designed to:
- Use only retrieved context
- Avoid hallucination
- Return fallback if no answer
  
**🧪 Example Queries**

“Which hotels have free WiFi and breakfast?”
“What is the cancellation policy of Hotel X?”
“Suggest a hotel near the beach with good reviews”.

**📊 Evaluation**

🔹 Metric Used: Precision@k
      Precision@k = Relevant Retrieved / k

**Approach:**
- Defined expected keywords per query
- Checked relevance of retrieved chunks

**📈 Sample Results**

| Query               | Precision@3 |
| ------------------- | ----------- |
| WiFi + Breakfast    | 0.67        |
| Beach + Reviews     | 0.67        |
| Cancellation Policy | 1.0         |

**🚫 Hallucination Control**

- Implemented via:

   Strict prompt:

      “Answer ONLY using the context”.

- Fallback response:

      “Not enough information available”.

- Context-limited input

**⚠️ Limitations**

- Keyword-based evaluation is simplistic
- Small synthetic dataset
- No semantic relevance scoring
- Model may miss implicit context

**🚀 Future Improvements**

- Use semantic similarity evaluation
- Larger real-world dataset
- Better chunking (sentence-based)
- Use advanced LLMs (e.g., GPT / LLaMA)
- Deploy as web app

**▶️ How to Run**

   - pip install sentence-transformers faiss-cpu transformers
   - python QA_System.py

**Modes**
🔹 Interactive Mode
 - Ask custom queries
 - Get answers with context
   
🔹 Evaluation Mode
 - Runs test queries
 - Shows Precision@k

**🧠 Key Learnings**

- Importance of retrieval quality in RAG
- Prompt engineering reduces hallucination
- Trade-off between model size and performance
- Evaluation is critical for system validation

**🙌 Conclusion**

This project demonstrates how RAG systems combine retrieval + generation to produce accurate, explainable answers from structured data.

**📎 Author**

**Sarvesh Kumar Shukla**
