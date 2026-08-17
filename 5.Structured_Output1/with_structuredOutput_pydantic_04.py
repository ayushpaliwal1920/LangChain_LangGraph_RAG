from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Initialize model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Schema definition
class Review(BaseModel):

    key_themes: list[str] = Field(
        description="Write all key themes"
    )

    summary: str = Field(
        description="Brief summary of review"
    )

    sentiment: Literal["pos++", "neg--"] = Field(
        description="Sentiment of review"
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="Pros in review"
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="Cons in review"
    )

    name: Optional[str] = Field(
        default=None,
        description="Reviewer name"
    )

# Structured output model
structured_model = model.with_structured_output(Review)

review_text = """
Amazon offers a wide range of services that are widely appreciated for their convenience, reliability, and innovation.
"""

# Invoke model
result = structured_model.invoke(review_text)

# Print complete object
print(result)

# Access fields correctly
print("\nSummary:", result.summary)
print("Sentiment:", result.sentiment)
print("Key Themes:", result.key_themes)
print("Pros:", result.pros)
print("Cons:", result.cons)
print("Reviewer Name:", result.name)