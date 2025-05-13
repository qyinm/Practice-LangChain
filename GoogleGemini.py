from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# setup the gemini pro
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

result = gemini_llm.invoke("What is the capital of France?")
print(result)