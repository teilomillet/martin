from config import Config
from llama_index.llms.litellm import litellm
from llama_index.core.llms import ChatMessage

message = ChatMessage(role="user", content="Hey! how's it going ?")

llm = LiteLLM("haiku")