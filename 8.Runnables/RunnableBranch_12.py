from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableLambda,RunnableParallel ,RunnablePassthrough ,RunnableSequence ,RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template= " Write a detailed report on {topic}",
    input_variables= ["topic"]
)


prompt2 = PromptTemplate(
    template = "Summarize the following text \n {text}",
    input_variables = ["text"]
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

report_gen_Chain = RunnableSequence(prompt1 , model , parser)

branchChain = RunnableBranch(
    (lambda x : len(x.split()) > 200 , RunnableSequence(prompt2, model , parser)), # if words > 500 : then summarize
    RunnablePassthrough()  # else(word < 500) print as it is 
)

finalChain = RunnableSequence(report_gen_Chain , branchChain)

result = finalChain.invoke({"topic" : "Russia vs Ukraine"})

print(result)