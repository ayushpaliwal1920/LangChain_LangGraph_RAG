from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model="gemini-2.5-flash")

# 1st prompt : detailed report

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt : summary

template2 = PromptTemplate(
    template='Write a 5 line summary on the following text.\n{text}',
    input_variables=['text']
)

# Generate report
prompt1 = template1.invoke({"topic": "black hole"})

result1 = model.invoke(prompt1)

# Generate summary
prompt2 = template2.invoke({"text": result1})

result2 = model.invoke(prompt2)

print("DETAILED REPORT:\n")
print(result1)

print("\nSUMMARY:\n")
print(result2)