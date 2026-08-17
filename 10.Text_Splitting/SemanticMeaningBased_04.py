from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text = """Cricket is one of the most popular sports in the world. It originated in England and is now played professionally in countries such as India, Australia, South Africa, and New Zealand. The sport consists of various formats including Test matches, One Day Internationals, and T20 games. The Indian Premier League is one of the richest cricket leagues globally.

Virat Kohli is considered one of the greatest batsmen of the modern era. He has scored thousands of international runs and captained the Indian cricket team across all formats. His fitness standards and consistency have inspired many young players.

Artificial Intelligence is transforming industries around the world. Machine learning algorithms can analyze large amounts of data and discover patterns that humans might miss. AI is widely used in healthcare, finance, transportation, and education.

Deep Learning is a subset of machine learning that uses neural networks with multiple layers. Applications include image recognition, speech recognition, autonomous vehicles, and natural language processing. Models such as transformers have revolutionized AI research.

The solar system consists of the Sun and all objects that orbit around it. Earth is the third planet from the Sun and the only known planet to support life. Mars is often called the Red Planet because of its reddish appearance.

Space exploration has advanced significantly over the last few decades. Missions to Mars, the Moon, and deep space have expanded our understanding of the universe. Private companies are also contributing to the development of reusable rockets and space tourism.

Healthy eating plays an important role in maintaining physical and mental well-being. A balanced diet includes fruits, vegetables, proteins, healthy fats, and whole grains. Proper nutrition helps prevent many chronic diseases.

Regular exercise improves cardiovascular health, strengthens muscles, and boosts mental health. Activities such as running, swimming, cycling, and strength training are commonly recommended by health professionals."""

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Semantic Chunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

docs = text_splitter.create_documents([text])

print("Number of chunks:", len(docs))

for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}")
    print(doc.page_content)
    print("-" * 50)