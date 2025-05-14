# Memory Caching
import os
from dotenv import load_dotenv
import time

if "GOOGLE_API_KEY" not in os.environ:
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from langchain_core.globals import set_llm_cache
from langchain_google_genai import GoogleGenerativeAI

# To make the caching really obvious, lets use a slower and older model.
# Caching supports newer chat models as well.
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer
start_wall = time.time()
start_cpu = time.process_time()

result = llm.invoke("Tell me a joke")

end_wall = time.time()
end_cpu = time.process_time()

print(result)
print(f"CPU time: {end_cpu - start_cpu:.2f} s")
print(f"Wall time: {end_wall - start_wall:.2f} s")

# The second time, it is in cache, so it should be faster
start_wall = time.time()
start_cpu = time.process_time()

result = llm.invoke("Tell me a joke")

end_wall = time.time()
end_cpu = time.process_time()

print(result)
print(f"CPU time: {end_cpu - start_cpu:.2f} s")
print(f"Wall time: {end_wall - start_wall:.2f} s")

