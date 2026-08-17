from langchain_openai import OpenAI 
from dotenv import load_dotenv  # It loads environment variables from a .env file into your Python program.

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')

result = llm.invoke("What is capital of India ?")

print(result)