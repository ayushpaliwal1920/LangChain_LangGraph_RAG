# It is a normal implemetation : basic chatbot 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 

# chat history :

chat_history = [
    SystemMessage(content="You are helpfull AI assistant")
]

while True:
    user_input = input("ME : ")

    chat_history.append(HumanMessage(content=user_input))# before model was not remebering history , so we decided to give it previous information also

    if user_input.strip().lower() == 'exit':
        break
    result = model.invoke(chat_history) # before model was not remebering history , so we decided to give it previous information also
    chat_history.append(AIMessage(content= result.content))
    print("AI : ", result.content)

print(chat_history)

 
# By following code we can check which  model/models do we have to use :

# import google.generativeai as genai
# genai.configure(api_key="YOUR_KEY")

# genai.configure(api_key="0909")

# for m in genai.list_models():
#     if "generateContent" in m.supported_generation_methods:
#         print(m.name)
