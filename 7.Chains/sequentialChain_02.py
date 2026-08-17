from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Prompt 1
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="""Generate a 5-point summary from the following text:{text}""",
    input_variables=["text"]
)

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Parser
parser = StrOutputParser()

# Chain
chain = prompt1 | model | parser | prompt2 | model | parser

# Invoke
result = chain.invoke({"topic": "How to get Internship in just 3 months"})

print(result)

# Visualize chain
print(chain.get_graph().print_ascii())