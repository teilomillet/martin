import kuzu
from llama_index.llms.litellm import LiteLLM
from config import Config
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.core import StorageContext, KnowledgeGraphIndex
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.tools import FunctionTool, ToolMetadata

# Initialize LiteLLM with API keys
LiteLLM.openai_key = Config.OPENAI_API_KEY
LiteLLM.anthropic_key = Config.ANTHROPIC_API_KEY
llm_model = LiteLLM(model="claude-3-haiku-20240307")

# Tool to load data from Wikipedia based on a list of topics.
def load_kg_data(topics):
    try:
        loader = WikipediaReader()
        documents = loader.load_data(pages=[topics], auto_suggest=False)
        return documents
    except Exception as e:
        print(f"Error loading data for topics {topics}: {str(e)}")
        return None

# Metadata for the load data tool
load_data_metadata = ToolMetadata(
    name="wiki_loader",
    description="Fetches and loads initial data from Wikipedia for specified topics, providing a broad overview suitable for preliminary information gathering."
)

# Register the load data function as a tool with metadata.
load_data_tool = FunctionTool(
    fn=load_kg_data,
    metadata=load_data_metadata
)

# Tool to build a knowledge graph from the loaded documents.
def build_kg(documents):
    try:
        db = kuzu.Database("test1")
        graph_store = KuzuGraphStore(db)
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=10,
            include_embeddings=True,
        )
        print(kg_index)
        return kg_index
    except Exception as e:
        print(f"Error building KG from documents: {str(e)}")
        return None

# Metadata for the build KG tool
build_kg_metadata = ToolMetadata(
    name="kg_builder",
    description="Constructs a detailed knowledge graph from loaded documents, enabling structured data analysis and supporting complex information queries."
)

# Register the build KG function as a tool with metadata.
build_kg_tool = FunctionTool(
    fn=build_kg,
    metadata=build_kg_metadata
)

# Tool to query the knowledge graph.
def query_kg(query, kg_index):
    try:
        query_engine = kg_index.as_query_engine(
            llm=llm_model,
            include_text=True,
            response_mode="tree_summarize",
            embedding_mode="hybrid",
            similarity_top_k=5,
            verbose=True,
        )
        print(query_engine.query(query))
        return query_engine.query(query)
    except Exception as e:
        print(f"Error querying KG: {str(e)}")
        return "Query failed"

# Metadata for the query KG tool
query_kg_metadata = ToolMetadata(
    name="kg_query",
    description="Performs detailed queries on a knowledge graph to extract specific information and insights, ideal for in-depth analysis and understanding complex relationships within the data."
)

# Register the query KG function as a tool with metadata.
query_kg_tool = FunctionTool(
    fn=query_kg,
    metadata=query_kg_metadata
)

kg_tools = [load_data_tool, build_kg_tool, query_kg_tool]