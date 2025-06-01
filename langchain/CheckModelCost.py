from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

prompt = "What is famous street foods in Seoul Korea in 50 characters?"
result = llm.invoke(prompt)
print("Response:", result)

with get_openai_callback() as callback:
    prompt = "What is famous street foods in Seoul Korea in 50 characters?"
    result = llm.invoke(prompt)
    # print("Response:", result)
    print("Total tokens:", callback.total_tokens)
