from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Parser
parser = StrOutputParser()

# Prompt
prompt = PromptTemplate(
    template="""
    Write a concise summary of the following text:

    {text}
    """,
    input_variables=["text"]
)

# Load document
loader = TextLoader("cricket.txt")
documents = loader.load()

print(documents)
print(documents[0].page_content)
print(documents[0].metadata)
print(type(documents))

# Chain
chain = prompt | model | parser

result = chain.invoke({
    "text": documents[0].page_content
})

print("\nSummary:")
print(result)