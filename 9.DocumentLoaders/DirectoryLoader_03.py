from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

#    "**/*.txt"  => All .txt files in a subfolder
#    "*.pdf"     => All .pdf files in the root directory 
#    "data/*.csv" => All .csv files in the folder
#    "**/*"      => All files
  

#  using load and lazt load fxn : 

loader = DirectoryLoader(
    path = "Book(test dirLoader)",
    glob= '*.pdf',
    loader_cls= PyPDFLoader
)

# docs = loader.load()

# for document in docs :
#     print(document.metadata)


docs = loader.lazy_load()  # when we have many documments then we use this 

for document in docs :
    print(document.metadata)