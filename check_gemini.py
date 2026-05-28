import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

def ping_gemini():
    print("📡 Testing connection to Google AI...")
    try:
        # Using a very lightweight model for the test
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=key)
        response = llm.invoke("Ping")
        print(f"✅ Success! Gemini said: {response.content}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    ping_gemini()
