from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# Define model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Define schema
schema = [
    ResponseSchema(
        name="fact_1",
        description="Fact 1 about the topic"
    ),

    ResponseSchema(
        name="fact_2",
        description="Fact 2 about the topic"
    ),

    ResponseSchema(
        name="fact_3",
        description="Fact 3 about the topic"
    ),

    ResponseSchema(
        name="fact_4",
        description="Fact 4 about the topic"
    )
]

# Create parser
parser = StructuredOutputParser.from_response_schemas(schema)

# Prompt template
template = PromptTemplate(
    template="""Give 4 facts about {topic}{format_instructions}""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

##### METHOD 1 : 

# # Create prompt
# prompt = template.invoke({
#     "topic": "Black hole"
# })

# # Invoke model
# result = model.invoke(prompt)

# # Parse output
# final_result = parser.parse(result.content)

# # Print final structured output
# print(final_result)

# # Access values
# print(final_result["fact_1"])
# print(final_result["fact_2"])
# print(final_result["fact_3"])
# print(final_result["fact_4"])



####### METHOD 2 : using chains

chain = template | model | parser
prompt = template.invoke({"topic" : 'black hole'})

result = chain.invoke(parser)

print(result)