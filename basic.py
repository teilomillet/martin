# basic.py
import fire
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.litellm import LiteLLM
from config import Config  # Check this import if running into issues

# __________________________________________________________________________________________________
# LLM

# Set API keys for LiteLLM
LiteLLM.openai_key = Config.OPENAI_API_KEY
LiteLLM.anthropic_key = Config.ANTHROPIC_API_KEY

# Initialize the LiteLLM model
llm = LiteLLM(model="claude-3-haiku-20240307")

# __________________________________________________________________________________________________
# Tools

# Define utility functions and wrap them as FunctionTools
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result."""
    return a + b

# Create tools from functions
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

# Collect all tools in a list
tools = [multiply_tool, add_tool]

# __________________________________________________________________________________________________
# Agent

# Initialize ReActAgent with tools and llm instance
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# __________________________________________________________________________________________________
# CLI

# Main function for CLI interaction
def main():
    while True:
        user_input = input("Please enter your message: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        response = agent.chat(user_input)
        print("Agent response:", response.response)

if __name__ == '__main__':
    fire.Fire(main)
