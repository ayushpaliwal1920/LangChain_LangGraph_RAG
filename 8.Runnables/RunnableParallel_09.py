from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel ,RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template= " Generate a tweet about {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= " Generate a linkdin post about {topic}",
    input_variables= ['topic']
)


model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

parallelChain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1 , model , parser),
    "linkedin" : RunnableSequence(prompt2 , model , parser)
})

result = parallelChain.invoke({"topic" : "Ai"})

print(result)

# print(result['tweet'])
# print(result['linkedin'])
