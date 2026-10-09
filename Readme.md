# RAG with LangChain

A small Retrieval-Augmented Generation (RAG) project built with LangChain. It ingests a set of PDF policy documents into a local Chroma vector store, then lets you chat with them from the terminal. Answers from OpenAI's `gpt-4o` are grounded in the retrieved chunks, and follow-up questions take the conversation history into account.

## How it works

1. **Ingestion** (`ingestion_pipeline.py`)
   - Reads every PDF in `docs/` page by page with `pypdf`, skipping empty pages. Each page becomes a LangChain `Document` with `source` and `page` metadata.
   - Splits the pages into chunks with `CharacterTextSplitter` (`chunk_size=100`, `chunk_overlap=0`).
   - Embeds the chunks with OpenAI `text-embedding-3-small` and persists them to Chroma at `db/chroma_db` (cosine similarity).

2. **History-aware chat** (`history_aware_generation.py`)
   - From the second question onward, `gpt-4o` first rewrites your question into a standalone search query, using the chat history.
   - Retrieves the top 3 matching chunks from Chroma and prints a preview of each.
   - Sends the question and the retrieved chunks to `gpt-4o` (`temperature=0.2`). If the answer isn't in the documents, the model is told to say so.
   - Appends each question and answer to the chat history.

## Tech stack

- Python 3.9+
- LangChain (`langchain-core`, `langchain-openai`, `langchain-chroma`, `langchain-text-splitters`)
- OpenAI: `text-embedding-3-small` for embeddings, `gpt-4o` for generation
- Chroma: a local, persistent vector store
- pypdf: PDF text extraction

## Project structure

```plaintext
Rag_p1/
├── docs/                         # Source PDFs (sample B-Cubed policy documents)
│   ├── b_cubed_data_security_policy.pdf
│   ├── b_cubed_engineering_policy.pdf
│   └── b_cubed_operations_leasing_policy.pdf
├── db/chroma_db/                 # Generated vector store (git-ignored)
├── ingestion_pipeline.py         # Load PDFs → chunk → embed → store in Chroma
├── history_aware_generation.py   # Interactive RAG chat with conversation memory
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for environment variables
└── Readme.md
```

## Getting started

### 1. Create a virtual environment

```bash
python3 -m venv virtual
source virtual/bin/activate   # On Windows: virtual\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

Run both scripts from the project root, because the vector store path `db/chroma_db` is relative to your current directory.

### Build the vector store

```bash
python ingestion_pipeline.py
```

To use your own documents, put PDFs in `docs/` and run this again. Running it again adds to the existing store instead of replacing it, so delete `db/chroma_db` first if you want a clean rebuild:

```bash
rm -rf db/chroma_db && python ingestion_pipeline.py
```

### Chat with your documents

```bash
python history_aware_generation.py
```

```text
Ask me anything! Type 'exit' to quit.

Your question: Which testing framework is the standard for front-end modules?
...
Your question: And what about backend services?
```

Type `exit` to quit.
