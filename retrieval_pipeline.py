from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata = {"hnsw:space": "cosine"}
)

query = "What is the official Document ID, effective date, and version of the B-Cubed engineering policy?"

query2 = "Which primary static analysis engine is designated as the unified tool for JavaScript and TypeScript?"

query3 = "Which testing framework is designated as the standard tool for front-end modules and decoupled backend services?"

retriever = db.as_retriever(search_kwargs={"k": 3})

# retriever.db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 3,
#          "score_threshold": 0.3}
#     )

relevant_docs = retriever.invoke(query3)
print(f"User query: {query3}")

print("-" * 100)

for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}: {doc.page_content}")
    print("-" * 100)


# combine the query and the relevant docs content
combined_input = f""" Based on the following Documents, Please answer this question:
{query2}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear helpful answer based on the documents provided. I you cant find the answer, say "I Don't have enough information to answer that question based on the information based on the provided documents.
"""

model = ChatOpenAI(model ="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content = combined_input)

]

result = model.invoke(messages)

print("\n ---- Generated response ---")

print("Content Only")
print(result.content)