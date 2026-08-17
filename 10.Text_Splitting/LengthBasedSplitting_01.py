from langchain.text_splitter import CharacterTextSplitter , TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# demo : text splitter

text = "Cricket is a popular sport played around the world. It has several formats including Test, ODI, and T20."

splitter = CharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 5,
    separator = ''
)

result = splitter.split_text(text)

print(result)



# workflow : document loader and text splitting 


loader = PyPDFLoader(
    "textPypdfLoader.pdf"
)

docs = loader.load()

result = splitter.split_documents(docs)  # for documents

print(result[0].page_content) # page content of 1st chunk
print(result[1].page_content) # page content of 2nd chunk

