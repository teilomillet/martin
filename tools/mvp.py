from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtracts second integer from first and returns the result."""
    return a - b

def divide(a: int, b: int) -> int:
    """Divides the first integer by the second and returns the result, handling division by zero."""
    if b == 0:
        return 0
    return a // b

# Create tools from functions
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)

# Collect all tools in a list
mvp_tools = [multiply_tool, add_tool, subtract_tool, divide_tool]
