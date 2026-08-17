from langchain_community.document_loaders import WebBaseLoader

from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Parser
parser = StrOutputParser()

# Prompt
prompt = PromptTemplate(
    template="""
    Answer the following question \n {question} from the following text- \n {text}
    """,
    input_variables=["text" , "question"]
)

url = "https://www.flipkart.com/peter-england-men-striped-casual-maroon-shirt/p/itm5e9ee7857744f?pid=SHTHDUABYZYVZA3Y&lid=LSTSHTHDUABYZYVZA3YYY4FO7&marketplace=FLIPKART&store=clo%2Fash&srno=b_1_4&otracker=browse&fm=organic&iid=9ebca78d-7f0b-4039-a37a-42e863e57d54.SHTHDUABYZYVZA3Y.SEARCH&ppt=None&ppn=None&ssid=hpbjnoy00g0000001780748226904&ov_redirect=true"

# url_list = [
#     url1 , url2 , url3
# ]

loader = WebBaseLoader(
    url
)

docs = loader.load()

chain = prompt | model | parser
result = chain.invoke({
    "question" : "what is the color of this shirt .",
    "text" : docs[0].page_content
})

print(result)