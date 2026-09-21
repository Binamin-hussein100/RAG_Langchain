# RAG with LangChain

A robust Retrieval-Augmented Generation (RAG) pipeline built using LangChain and Python. This repository demonstrates how to ingest private documents, embed them into a vector space, retrieve relevant context, and leverage Large Language Models (LLMs) to generate accurate, context-grounded responses.

## 🚀 Features

- Document Ingestion & Splitting: Load and split text/PDF documents efficiently using LangChain text splitters.
- Embeddings & Vector Search: Convert text into vector embeddings and index them using a vector database for high-performance similarity search.
- Contextual Retrieval: Retrieve the most relevant chunks of information based on user queries.
- LLM Integration: Chain retrieved documents with an LLM to generate precise and grounded answers, minimizing hallucinations.

## 🛠️ Tech Stack

- Python (3.10+)
- LangChain (Orchestration framework)
- OpenAI / HuggingFace (LLMs & Embeddings)
- Chroma / FAISS (Vector Store — update if using a different DB)

## 📂 Project Structure

```plaintext
RAG_Langchain/
│
├── data/               # Source documents (PDFs, TXT, etc.)
├── notebooks/          # Interactive Jupyter Notebooks (if applicable)
├── src/                # Source code modules
│   ├── ingest.py       # Document loading and embedding pipeline
│   └── rag_chain.py    # Retrieval and generation logic
├── .env.example        # Template for environment variables
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── LICENSE             # License information
```

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Binamin-hussein100/RAG_Langchain.git
cd RAG_Langchain
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory by copying the example template:

```bash
cp .env.example .env
```

Add your API keys inside `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 💡 Usage

Run the main script to start querying your documents:

```bash
python src/rag_chain.py
```

Alternatively, if your project is notebook-based, launch Jupyter Notebook and open the workflow notebooks:

```bash
jupyter notebook
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
