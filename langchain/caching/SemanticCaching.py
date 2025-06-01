from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.globals import set_llm_cache
from langchain_redis import RedisSemanticCache

from dotenv import load_dotenv
import os
import time

# 프롬프트 간단한 정규화
def normalize_prompt(prompt):
    prompt = prompt.lower()
    prompt = prompt.strip()
    return prompt

# 캐시에서 가져온 응답 검증
def validate_cached_response(response, required_items=5):
    items = response.split("\n")
    if len(items) >= required_items:
        return True
    else:
        return False

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07"
)

redis_cache = RedisSemanticCache(
    redis_url=f"redis://default:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}",
    embeddings=embeddings,
    distance_threshold=0.3
)

set_llm_cache(redis_cache)

# prompt1 = "What is top 10 famous street foods in Paris France in 200 characters?"
# prompt2 = "What is top 5 famous street foods in Paris France in 200 characters?"
prompt1 = normalize_prompt("What is top 10 famous street foods in Paris France in 200 characters?")
prompt2 = normalize_prompt("What is top 5 famous street foods in Paris France in 200 characters?")

start_time = time.time()
result = llm.invoke(prompt1)
print(result)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

print("=" * 20)

start_time = time.time()
result = llm.invoke(prompt2)
print(result)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")