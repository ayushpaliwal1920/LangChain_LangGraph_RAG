from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = GoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Schema
class Person(BaseModel):
    name: str = Field(description="Name of fictional person")
    age: int = Field(description="Age of person")
    city: str = Field(description="City of person")

# Parser
parser = JsonOutputParser(
    pydantic_object=Person
)

# Prompt
template1 = PromptTemplate(
    template=""" Generate a fictional person.{format_instruction}""",

    input_variables=[],

    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

#### Method 1 : 

# prompt = template1.format()

# # GoogleGenerativeAI returns STRING
# result = model.invoke(prompt)

# # Parse JSON string
# final_result = parser.parse(result)

# print(final_result)
# print(type(final_result))



### OR ### : Method 2  :  use chain 

chain = template1 | model | parser

result = chain.invoke({})

print(result)



