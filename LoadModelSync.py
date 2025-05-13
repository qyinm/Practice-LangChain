# Invoke
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

prompt = "What is famous street foods in Seoul Korea in 200 characters?"
llm.invoke(prompt)

# Batch 호출
prompts = [
    "What is famous street foods in Seoul Korea in 200 characters?",
    "What is famous street foods in Tokyo Japan in 200 characters?",
    "What is famous street foods in Bangkok Thailand in 200 characters?",
]
responses = llm.batch(prompts)

print(responses)
print('='*10)
# 스트리밍 호출
prompt = "What is famous street foods in Seoul Korea in 200 characters?"
for chunk in llm.stream(prompt):
    print(chunk, flush=True)