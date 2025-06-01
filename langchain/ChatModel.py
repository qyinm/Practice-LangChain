from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

messages = [
    SystemMessage(content="You are the travel agent. You can provide travel itinery to the user."),
    HumanMessage(content="Where is the top 3 popular space for tourist in Seoul?")
]

aiMessage = chat.invoke(messages)
print(aiMessage.content)

# AIMessage 추가
messages.append(aiMessage)
print("-"*30)

# 새로운 대화 추가
messages.append(HumanMessage(content="Which transport can I use to visit the places?"))

aiMessage = chat.invoke(messages)
print(aiMessage.content)
print("-"*30)

# 새로운 대화 추가
messages.append(HumanMessage(content="Where is the good restaurant for family near the place?"))

aiMessage = chat.invoke(messages)
print(aiMessage.content)

# ChatMessageHistory는 체인 외부에서 직접 메모리를 관리하는 경우에 사용됨.
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("Where is the top 3 popular space for tourist in Seoul?")
aiMessage = chat.invoke(history.messages)
history.add_ai_message(aiMessage)
print(aiMessage.content)
print("-"*30)

history.add_user_message("Which transport can I use to visit the places?")
aiMessage = chat.invoke(history.messages)
history.add_ai_message(aiMessage.content)
print(aiMessage.content)
print("-"*30)

