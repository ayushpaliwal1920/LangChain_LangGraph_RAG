from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

# Initialize model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Schema definition
class Review(TypedDict):
    summary: str
    sentiment: Literal["pos", "neg"]
    emotion: Annotated[str, "Emotion or vibe of the review"]
    key_themes: Annotated[list[str], "Key themes discussed in the review"]
    pros: Annotated[Optional[list[str]], "List of pros"]
    cons: Annotated[Optional[list[str]], "List of cons"]
    name: Annotated[Optional[str], "Name of the reviewer"]

# Structured output model
structured_model = model.with_structured_output(Review)

review_text = """
Amazon offers a wide range of services that are widely appreciated for their convenience, reliability, and innovation. Its e-commerce platform provides fast delivery, competitive pricing, and an enormous selection of products, making online shopping simple and efficient for customers around the world. The company’s cloud computing division, Amazon Web Services, is especially popular among developers and businesses because of its scalable infrastructure, powerful tools, and strong performance. Services like streaming, digital payments, and smart devices also contribute to Amazon’s strong ecosystem. While some users mention concerns about customer support consistency or pricing changes in certain services, Amazon continues to remain one of the most trusted and influential technology companies globally due to its customer-focused approach and continuous innovation.
"""

# Invoke model
result = structured_model.invoke(review_text)

# Print structured response
print(result)

print("\nSummary:", result["summary"])
print("Sentiment:", result["sentiment"])
print("Emotion:", result["emotion"])
print("Key Themes:", result["key_themes"])
print("Pros:", result["pros"])
print("Cons:", result["cons"])
print("Reviewer Name:", result["name"])