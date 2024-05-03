# basicv2.py
import fire
from llama_index.core.agent import ReActAgent
from llama_index.llms.litellm import LiteLLM
from llama_index.core.llms import ChatMessage
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

# Initialize ReActAgent with tools and llm instance
agent = ReActAgent.from_tools(mvp_tools, llm=llm, verbose=True)

# __________________________________________________________________________________________________
# CLI

# Main function for CLI interaction
def main():
    while True:
        user_input = input("Please enter your message: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        # System message
        system_message = ChatMessage(role="system", content="You are a pirate from the 18th.")

        # User message
        user_message = ChatMessage(role="user", content=user_input)

        # Combine messages for conversation
        messages = [system_message, user_message]

        # Get response from the model
        response = llm.chat(messages)
        print("Agent response:", response)

if __name__ == '__main__':
    fire.Fire(main)
