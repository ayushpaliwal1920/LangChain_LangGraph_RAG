from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# Models
model1 = ChatGoogleGenerativeAI( model="gemini-2.5-flash")

model2 = ChatGoogleGenerativeAI( model="models/gemini-2.0-flash-001")

# Prompt 1 → Notes
prompt1 = PromptTemplate(
    template="""Generate short and simple notes from the following text:{text}""",
    input_variables=["text"]
)

# Prompt 2 → Quiz
prompt2 = PromptTemplate(
    template="""Generate 5 short questions from the following text:{text}""",
    input_variables=["text"]
)

# Prompt 3 → Merge
prompt3 = PromptTemplate(
    template="""Merge the provided notes and quiz into a single document.Notes:{notes} Quiz:{quiz}""",
    input_variables=["notes", "quiz"]
)

# Parser
parser = StrOutputParser()

# Parallel Chain
parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)

# Merge Chain
merge_chain = prompt3 | model1 | parser

# Final Chain
chain = parallel_chain | merge_chain

# Input Text
text = """
Artificial Intelligence is transforming industries by enabling machines
to learn from data and make intelligent decisions.
"""

# Invoke
result = chain.invoke({"text": text})

print(result)

print(chain.get_graph().print_ascii())