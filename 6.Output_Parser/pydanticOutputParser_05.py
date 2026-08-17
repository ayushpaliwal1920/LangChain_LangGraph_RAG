from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Define schema
class Person(BaseModel):

    name: str = Field(
        description="Name of the person"
    )

    age: int = Field(
        gt=18,
        lt=100,
        description="Age of the person"
    )

    city: str = Field(
        description="Name of the city of the person"
    )

# Parser
parser = PydanticOutputParser(
    pydantic_object=Person
)

# Prompt
template = PromptTemplate(
    template="""
    Generate the name, age and city
    of a fictional {place} person.
    {format_instructions}
    """,

    input_variables=["place"],

    partial_variables={
        "format_instructions":
        parser.get_format_instructions()
    }
)

####### METHOD 1 : ######

# # Create prompt
# prompt = template.invoke({
#     "place": "Indian"
# })

# # Invoke model
# result = model.invoke(prompt)

# # Parse output
# final_result = parser.parse(result.content)

# # Print result
# print(final_result)

# print(final_result.name)
# print(final_result.age)
# print(final_result.city)

####### METHOD 2 : ######

chain = template | model | parser

final_result = chain.invoke({"place" : "sri lanka"})

print(final_result) 