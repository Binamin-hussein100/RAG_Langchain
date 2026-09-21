from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

persist_directory = "db/chroma_db"
embedding = OpenAIEmbeddings(model = "text-embedding-3-small")

db = Chroma(persist_directory=persist_directory, embedding_function=embedding)


model = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.2, 
    max_tokens=2000
    )

chat_history = []


def ask_question(user_question):
    print(f"\n --- You asked: {user_question} ---\n")
    if chat_history:
        messages = [
            SystemMessage(content= "Given the chat history, rewrite the new question to be a standalone question. Then answer the standalone and searchable. Just return the rewritten question"),
        ] + chat_history + [HumanMessage(content=f"New question: {user_question}")]

        result = model.invoke(messages)
        search_question = result.content.strip()
    else:
        search_question = user_question    


    #  find relevant question
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents for the question: {search_question}")
    for i, doc in enumerate(docs,1):
        lines = doc.page_content.splitlines()
        preview = "\n".join(lines[:5])  # Show first 5 lines as a preview
        print(f"\n--- Document {i} Preview ---\n{preview}\n---" + "-" * 30)


# combine the query and the relevant docs content
    combined_input = f""" Based on the following Documents, Please answer this question:
    {user_question}

    Documents:
    {chr(10).join([f"-{doc.page_content}" for doc in docs])}

    Please provide a clear helpful answer based on the documents provided. I you cant find the answer, say "I Don't have enough information to answer that question based on the information based on the provided documents.
    """

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents and conversation histories. If the answer is not found in the documents, respond with 'I don't have enough information to answer that question based on the provided documents.'"),
        HumanMessage(content = combined_input)

    ]

    result = model.invoke(messages)
    answer = result.content

    # Remeber conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print("Answer:\n", answer)
    return answer


def start_chat():
    print("Ask me anything! Type 'exit' to quit.")

    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break
        ask_question(user_input)


if __name__ == "__main__":
    start_chat()



