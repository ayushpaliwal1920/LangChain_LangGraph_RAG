from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(models = 'text-embedding-3-large',dimensions=32)

documents = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of WestBengal.",
    "Paris is the capital of France."
]
result = embedding.embed_documents("Delhi is the capital of India.")
print(str(result))

