import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document

dog = [12,123,34]
load_dotenv()

def load_documents(folder_path="docs"):
    print(f"Loading documents from {folder_path}...")
    # Resolve relative to the current file if it's a relative path, or use it as-is
    if not os.path.isabs(folder_path):
        target_dir = os.path.join(os.path.dirname(__file__), folder_path)
    else:
        target_dir = folder_path
        
    documents = []
    
    for file in os.listdir(target_dir):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(target_dir, file))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                
                doc = Document(
                    page_content=text,
                    metadata={"source": file, "page": page_number}
                )
                documents.append(doc)
    # print(f"Loaded {len(documents)} pages into LangChain Document objects from {folder_path}")
    return documents

def split_documents(documents,chunk_size=100,chunk_overlap=0):
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
         chunk_overlap=chunk_overlap
         )
    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n ---chunk {i+1}---")
            print(f"length: {len(chunk.page_content)} characters")
            print(chunk.page_content)
            print(chunk.metadata)
            print("\n")
    else:
        print("No chunks found")
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist chromadb vector store from the chunks"""
    print("Creating embeddings and storing in chromadb vector store...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("finished creating vector store")
    print("Vector store created and stored in chromadb vector store")
    return vector_store

def main():
    print("Starting ingestion pipeline...") 
    # load documents
    documents = load_documents()
    # split documents into chunks
    chunks = split_documents(documents)
    #create vector store
    vector_store = create_vector_store(chunks)

    #query vector store
   


    
if __name__ == "__main__":
    main()