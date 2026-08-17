from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

documents = [
    "Football is widely played in Europe.",
    "Artificial Intelligence is transforming technology.",
    "Machine learning is a subset of Artificial Intelligence.",
    "Deep learning models require large amounts of data.",
    "Python is a popular programming language for AI.",
    "Java is commonly used for enterprise applications.",
    "Data science involves statistics and programming.",
    "Natural Language Processing deals with text data.",
    "Transformers are powerful models for NLP tasks.",
    "The Taj Mahal is located in Agra.",
    "The Eiffel Tower is located in Paris.",
    "Mount Everest is the tallest mountain in the world.",
    "The Amazon rainforest contains diverse wildlife."
]

embedding = OpenAIEmbeddings(model = "text-embedding-3-large",dimensions=300)
query = 'tell me about machine learning'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding] , doc_embeddings)[0]

index , score = sorted(list(enumerate(scores)),key=lambda x : x[1])[-1]

print(query)
print(documents[index])
print("similarity score is : ",score)


