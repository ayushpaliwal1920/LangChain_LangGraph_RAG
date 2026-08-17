from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

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

# Create vector store
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# Create retriever
retriever = vectorstore.as_retriever()

# Gemini LLM
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Query
query = "What are the key takeaways from the document?"

# Run chain
answer = qa_chain.invoke({"query": query})

print(answer["result"])