# # from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
# # from dotenv import load_dotenv
# # import os

# # load_dotenv()

# # llm = HuggingFaceEndpoint(
# #     repo_id="HuggingFaceH4/zephyr-7b-beta",
# #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
# #     task="conversational"
# # )

# # model = ChatHuggingFace(llm = llm)

# # result = llm.invoke("What is openSource and closeSource ?")
# # print(result.content)

# from dotenv import load_dotenv
# import os

# from langchain_huggingface import (
#     HuggingFaceEndpoint,
#     ChatHuggingFace
# )

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="conversational",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# )

# chat_model = ChatHuggingFace(llm=llm)

# result = chat_model.invoke(
#     "What is open source and closed source?"
# )

# print(result.content)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)
result = chat_model.invoke("Hello, how are you?")
print(result.content)