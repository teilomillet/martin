import kuzu
import fire
from llama_index.llms.litellm import LiteLLM
from config import Config
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.core import StorageContext, KnowledgeGraphIndex
from llama_index.readers.wikipedia import WikipediaReader


# __________________________________________________________________________________________________
# LLM

# Set API keys for LiteLLM
LiteLLM.openai_key = Config.OPENAI_API_KEY
LiteLLM.anthropic_key = Config.ANTHROPIC_API_KEY

# Initialize the LiteLLM model
llm = LiteLLM(model="claude-3-haiku-20240307")

# Build the Kuzu db
db = kuzu.Database("test1")
graph_store = KuzuGraphStore(db)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

loader = WikipediaReader()

documents = loader.load_data(
    pages=["Graph theory"], auto_suggest=False
)

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    include_embeddings=True,
    verbose=True,
)


rel_map = graph_store.get_rel_map()

# query using top 3 triplets plus keywords (duplicate triplets are removed)
query_engine = kg_index.as_query_engine(
    llm=llm,
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
    verbose=True,
)


# Main function for CLI interaction
def main():
    while True:
        user_input = input("Please enter your message: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        # Get response from the model
        response = query_engine.query(user_input)
        print("Agent response:", response)

if __name__ == '__main__':
    fire.Fire(main)
