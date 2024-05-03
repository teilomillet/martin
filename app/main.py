from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from llama_index.core.agent import ReActAgent
from llama_index.llms.litellm import LiteLLM
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from config import Config
from tools.mvp import mvp_tools
from tools.kg import kg_tools

# Initialize FastAPI
app = FastAPI()

# Initialize the chat store and memory buffer for maintaining conversation history
chat_store = SimpleChatStore()
chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=10000,  # Maximum number of tokens that can be stored
    chat_store=chat_store,
    chat_store_key="global"  # A global key for session management; consider dynamic keys for different users
)

# Configure API keys for LiteLLM from environment variables or configuration files
LiteLLM.openai_key = Config.OPENAI_API_KEY
LiteLLM.anthropic_key = Config.ANTHROPIC_API_KEY

# Initialize the LLM with a specific model configuration
llm = LiteLLM(model="claude-3-haiku-20240307")

# Initialize the ReActAgent with specific tools and the LLM instance
agent = ReActAgent.from_tools(
    tools=mvp_tools + kg_tools,
    llm=llm,
    verbose=True,
    chat_history=ChatMessage,
    memory=chat_memory,
    context="You are a pirate from the 18th century. Speak accordingly."
)

# Set up Jinja2 for HTML template rendering
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Serve the initial page with empty chat history or a welcome message
    return templates.TemplateResponse("chat.html", {"request": request, "history": []})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    # Process the user input and generate a response using the LLM
    response = agent.chat(user_input)
    # Retrieve the updated conversation history from memory
    history = chat_store.get_messages(key="global")
    print(history)
    # Return the chat page with updated history to display conversation
    return templates.TemplateResponse("chat.html", {"request": request, "history": history})

if __name__ == "__main__":
    import uvicorn
    # Start the application with Uvicorn with default settings; adjust as necessary for deployment
    uvicorn.run(app, host="127.0.0.1", port=8000)
