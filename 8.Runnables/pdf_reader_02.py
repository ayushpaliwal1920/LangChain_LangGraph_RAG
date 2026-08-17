from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load document
loader = TextLoader("docs.txt")
documents = loader.load()

# Split document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Store vectors in FAISS
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# Create retriever
retriever = vectorstore.as_retriever()

# Query
query = "What are the key takeaways from the document?"

# Retrieve relevant chunks
retrieved_docs = retriever.invoke(query)

# Combine retrieved text
retrieved_text = "\n".join(
    [doc.page_content for doc in retrieved_docs]
)

# LLM
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# RAG prompt
prompt = f"""
Based on the following context, answer the question.

Context:
{retrieved_text}

Question:
{query}
"""

# Generate answer
answer = llm.invoke(prompt)

print("Answer:")
print(answer)