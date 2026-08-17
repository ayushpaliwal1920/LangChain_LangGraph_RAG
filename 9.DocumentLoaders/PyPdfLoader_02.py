from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader # for simple text data 
from langchain_community.document_loaders import PDFPlumberLoader # for pdf with tables and columns
from langchain_community.document_loaders import UnstructuredPDFLoader , AmazonTextractPDFLoader # for pdf with scanned images
from langchain_community.document_loaders import PyMuPDFLoader # need image and layout data
from langchain_community.document_loaders import UnstructuredPDFLoader # want best structure extraction

# and many more pdfloaders

load_dotenv()

loader = PyPDFLoader(
    "textPypdfLoader.pdf",
)

docs = loader.load()

print(len(docs)) # pages of documents

print(docs[0].page_content) # content of first page
print(docs[0].metadata)     # meta data of first page
