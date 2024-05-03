# main.py
import fire
from llama_index.core.agent import ReActAgent
from llama_index.llms.litellm import LiteLLM
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

from config import Config  # Check this import if running into issues
from tools.mvp import mvp_tools  # Import tools from mvp.py

# __________________________________________________________________________________________________
# LLM

# Set API keys for LiteLLM
LiteLLM.openai_key = Config.OPENAI_API_KEY
LiteLLM.anthropic_key = Config.ANTHROPIC_API_KEY

# Initialize the LiteLLM model
llm = LiteLLM(model="claude-3-haiku-20240307")

# __________________________________________________________________________________________________
# Agent

# Add memory
memory = ChatMemoryBuffer.from_defaults(token_limit=10000)

# Initialize ReActAgent with tools and llm instance
agent = ReActAgent.from_tools(
    mvp_tools, 
    llm=llm, 
    verbose=True, 
    chat_history=ChatMessage, 
    memory=memory, 
    context="You are a pirate from the 18th. So talk like it."
    )

# __________________________________________________________________________________________________
# CLI

# Main function for CLI interaction
def main():
    while True:
        user_input = input("Please enter your message: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        # Get response from the model
        response = agent.chat(user_input)
        print("Agent response:", response)

if __name__ == '__main__':
    fire.Fire(main)
