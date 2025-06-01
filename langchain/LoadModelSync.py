# Invoke
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# 단일 호출
prompt = "What is famous street foods in Seoul Korea in 200 characters?"
response = llm.invoke(prompt)
print("단일 호출 결과:", response)
print('='*10)

# Batch 호출 (순차적 처리)
prompts = [
    "What is famous street foods in Seoul Korea in 200 characters?",
    "What is famous street foods in Tokyo Japan in 200 characters?",
    "What is famous street foods in Bangkok Thailand in 200 characters?",
]

print("배치 호출 결과:")
for prompt in prompts:
    response = llm.invoke(prompt)
    print(response)
    time.sleep(4)  # 4초 대기

print('='*10)

# 스트리밍 호출
prompt = "What is famous street foods in Seoul Korea in 200 characters?"
print("스트리밍 호출 결과:")
for chunk in llm.stream(prompt):
    print(chunk, flush=True)