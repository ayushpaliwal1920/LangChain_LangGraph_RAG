from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableLambda,RunnableParallel ,RunnablePassthrough ,RunnableSequence

load_dotenv()

# Demo of Runnable Lambda : 

def word_counter(text) :
    return len(text.split())

runnable_wordCount = RunnableLambda(word_counter)

print(runnable_wordCount.invoke("Hi How are you ?"))

# Example  : joke word counter 

prompt = PromptTemplate(
    template= " Joke about the following {topic}",
    input_variables= ['topic']
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt , model , parser)

parallelChain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "wordCount" : RunnableLambda(
        lambda x : len(x.split())
    )
})

finalChain = RunnableSequence(joke_gen_chain , parallelChain)

result = finalChain.invoke({'topic' : "Ai"})

final_result = """{} \n word Count - {}""".format(result['joke'] , result['wordCount'])
