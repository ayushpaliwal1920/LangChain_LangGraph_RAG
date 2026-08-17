from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel ,RunnablePassthrough ,RunnableSequence

load_dotenv()

# demo of passthrough : 

passthrough = RunnablePassthrough()
print(passthrough.invoke({'name' : "Ayush"})) # returns input as output without modifying 
print("\n")

# example : 

prompt1 = PromptTemplate(
    template= " Joke about the following {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= " explain the following joke : {text}",
    input_variables= ['text']
)


model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

jokeGenChain = RunnableSequence(prompt1 , model , parser)

parallelChain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "Explaination" : RunnableSequence(prompt2 , model , parser)
})


finalChain = RunnableSequence(jokeGenChain , parallelChain)

result = finalChain.invoke({"topic" : "Cricket"})

print(result)