from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# Prompt :

prompt = PromptTemplate(
            template = "Generate 5 interesting facts about {topic} ",
            input_variables= ['topic'],

)

#  Model :

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


# Parser :

parser  = StrOutputParser()

# Chain :

chain = prompt | model | parser

result = chain.invoke({"topic" : "cricket"})

print(result)

print(chain.get_graph().print_ascii())