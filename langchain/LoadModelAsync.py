# 비동기 호출예제
import asyncio
import time
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
prompt = "What is famous street foods in Seoul Korea in 50 characters?"

# 비동기 호출
async def invoke_async(llm):
    result = await llm.ainvoke(prompt)
    return result

async def invoke_parallel():
    tasks = [invoke_async(llm) for _ in range(10)]
    await asyncio.gather(*tasks)

start_time = time.perf_counter()
asyncio.run(invoke_parallel())
end_time = time.perf_counter()
print(f"비동기 호출 시간: {end_time - start_time}초")

# 동기 호출
start_time = time.perf_counter()
for _ in range(10):
    result = llm.invoke(prompt)
    print(result)
end_time = time.perf_counter()
print(f"동기 호출 시간: {end_time - start_time}초")


    