from langchain_google_genai import GoogleGenerativeAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate

# Load the LLM (GPT - 3.5)

llm = GoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.4)

# Create a Prompt Template 

prompt = PromptTemplate(
    input_variables=['topic'] , 
    template="Suggest a catchy blog title about {topic}."
)

# create a llmchain

chain = LLMChain(llm = llm , prompt = prompt)

# Run the chain with a specific topic

topic = input("Enter a topic")
output = chain.run(topic)

print("Generated Blog title :" , output)