from langchain_google_genai import GoogleGenerativeAI
from langchain.globals import set_llm_cache
from langchain_redis import RedisCache
from redis import Redis

import os
from dotenv import load_dotenv
import time

load_dotenv()

redis_cache = RedisCache(redis_client=Redis(
    host=os.getenv("REDIS_HOST"),
    port=15548,
    decode_responses=True,
    username="default",
    password=os.getenv("REDIS_PASSWORD"),
))
set_llm_cache(redis_cache)

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
)
prompt = "What is famous singer in Seoul Korea in 30 characters?"

print("첫 번째 LLM 호출:")
start_time = time.time()
reslut = llm.invoke(prompt)
end_time = time.time()
print(reslut)
print(f"Time taken: {end_time - start_time:.2f} seconds")

print("\n" + "=" * 20)
print("두 번째 LLM 호출 (캐시 확인):")

# 두 번째 호출도 동일하게 핸들러 전달
start_time = time.time()
reslut = llm.invoke(prompt)
end_time = time.time()
print(reslut)
print(f"Time taken: {end_time - start_time:.2f} seconds")
