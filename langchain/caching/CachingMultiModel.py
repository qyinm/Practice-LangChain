from langchain_core.caches import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_google_genai import GoogleGenerativeAI
from langchain.globals import set_debug
import os
from dotenv import load_dotenv
import time

set_debug(True)

if "GOOGLE_API_KEY" not in os.environ:
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

set_debug(False)

set_llm_cache(InMemoryCache())

llm1 = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
llm2 = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, cache=False)

start_wall = time.time()
start_cpu = time.process_time()

llm1.invoke("Tell me a joke")

end_wall = time.time()
end_cpu = time.process_time()

print(f"CPU time: {end_cpu - start_cpu:.2f} s")
print(f"Wall time: {end_wall - start_wall:.2f} s")

start_wall = time.time()
start_cpu = time.process_time()

llm2.invoke("Tell me a joke")

end_wall = time.time()
end_cpu = time.process_time()

print(f"CPU time: {end_cpu - start_cpu:.2f} s")
print(f"Wall time: {end_wall - start_wall:.2f} s")



