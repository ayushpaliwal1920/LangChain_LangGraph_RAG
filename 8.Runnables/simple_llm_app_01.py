from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Init LLm

llm = GoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.7)

# create prompt template 

prompt = PromptTemplate(
    input_variables=["topic"],
    template= "Suggest a catchy blog title about {topic}"   
)

# define the input :

topic = input("Enter a topic")

# Format the prompt manually using Prompt topic :

formatted_topic = prompt.format(topic = topic)

# call the llm 

blog_title = llm.predict(formatted_topic)

# print the output :

print("Generate blog title :" , blog_title)