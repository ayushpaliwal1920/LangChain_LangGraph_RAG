from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

# Initialize model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Schema definition :

Review = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "title": "Key Themes",
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write all key themes"
    },

    "summary": {
      "title": "Summary",
      "type": "string",
      "description": "Brief summary of review"
    },

    "sentiment": {
      "title": "Sentiment",
      "type": "string",
      "enum": ["pos++", "neg--"],
      "description": "Sentiment of review"
    },

    "pros": {
      "title": "Pros",
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Pros in review"
    },

    "cons": {
      "title": "Cons",
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Cons in review"
    },

    "name": {
      "title": "Name",
      "type": ["string", "null"],
      "description": "Reviewer name"
    }
  },

  "required": [
    "key_themes",
    "summary",
    "sentiment"
  ]
}

# Structured output model
structured_model = model.with_structured_output(Review)

review_text = """
Amazon offers a wide range of services that are widely appreciated for their convenience, reliability, and innovation.
"""

# Invoke model
result = structured_model.invoke(review_text)

# Print complete object
print(result)

# Access like dictionary
print("\nSummary:", result["summary"])
print("Sentiment:", result["sentiment"])
print("Key Themes:", result["key_themes"])
print("Pros:", result["pros"])
print("Cons:", result["cons"])
print("Reviewer Name:", result["name"])