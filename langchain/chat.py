from langchain.chains import SingleLM
from langchain.applications import Chat

# Initialize your model
gpt2_model = MyGPT2Model()

# Use LangChain's SingleLM for conversation
conversation_agent = SingleLM(gpt2_model)

# Create a Chat application
chat = Chat(conversation_agent)

# Example conversation
response = chat.respond_to("Hello, how are you?")
print(response)

from fastapi import FastAPI

app = FastAPI()

@app.post("/chat/")
async def chat_endpoint(user_input: str):
    response = chat.respond_to(user_input)
    return {"response": response}
