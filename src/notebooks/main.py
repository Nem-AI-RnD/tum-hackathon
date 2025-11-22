# %% [markdown]
# ## LlamaIndex + Jina AI + Qdrant RAG Pipeline Setup

# %%

import os
from typing import List, Any
import qdrant_client
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_anthropic import ChatAnthropic

from ai_eval.resources import deepeval_scorer as deep
from ai_eval.resources.rag_template import RAG
from ai_eval.resources import eval_dataset_builder as eval_builder
from ai_eval.services.file import JSONService
from ai_eval.config import global_config as glob
from langchain_community.document_loaders import PyPDFLoader

from llama_index.core import (
    Document as LlamaIndexDocument,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.langchain import LangChainLLM

# Load environment variables from .env file
load_dotenv()

# 1. Load and validate required API keys
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError(
        "ANTHROPIC_API_KEY not found in environment. Please set it in your .env file.")

jina_api_key = os.getenv("JINA_API_KEY")
if not jina_api_key:
    raise ValueError(
        "JINA_API_KEY not found in environment. Please set it in your .env file.")

qdrant_api_key = os.getenv("QDRANT_API_KEY")
if not qdrant_api_key:
    raise ValueError(
        "QDRANT_API_KEY not found in environment. Please set it in your .env file.")

qdrant_url = os.getenv("QDRANT_URL")
if not qdrant_url:
    raise ValueError(
        "QDRANT_URL not found in environment. Please set it in your .env file.")

# Optional: LlamaCloud API key (if using LlamaCloud features)
llamacloud_api_key = os.getenv("LLAMACLOUD_API_KEY")
if llamacloud_api_key:
    os.environ["LLAMA_CLOUD_API_KEY"] = llamacloud_api_key

print("✓ All required API keys loaded successfully")

# 2. Configure Jina AI embeddings for passage encoding (for document nodes)

# Configure global embedding model (Jina) for both passages and queries
# (simpler and fully supported by LlamaIndex; you can later split into
# retrieval.passage / retrieval.query if needed with a custom service context).
embed_model = JinaEmbedding(
    api_key=jina_api_key,
    model="jina-embeddings-v3",
    task="retrieval.passage",  # used for both docs & queries in this example
)

Settings.embed_model = embed_model

# 2. Load documents directly using PyPDFLoader (skip Preprocessor to avoid multiprocessing issues)
filename = "Allplan_2020_Manual.pdf"

loader = PyPDFLoader(f"{glob.DATA_PKG_DIR}/{filename}")
raw_docs = loader.load()

print(f"Loaded {len(raw_docs)} pages from PDF")

# Convert LangChain Documents to LlamaIndex Documents directly
llama_documents = [
    LlamaIndexDocument(
        text=doc.page_content,
        metadata={
            **(doc.metadata if hasattr(doc, "metadata") and doc.metadata else {}),
            "source": filename,
            "page": i + 1,
        }
    )
    for i, doc in enumerate(raw_docs)
]

print(f"Converted {len(llama_documents)} pages to LlamaIndex format")

# 3. Chunk documents into nodes using LlamaIndex SentenceSplitter (no multiprocessing)
parser = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=200,
)
nodes = parser.get_nodes_from_documents(llama_documents)

print(f"Created {len(nodes)} nodes from {len(llama_documents)} documents")

# Convert back to LangChain Documents for RAG class compatibility (optional, not used when index/retriever are provided)
documents = [
    Document(
        page_content=doc.text,
        metadata=doc.metadata
    )
    for doc in llama_documents
]

# %%

# 4. Connect to Qdrant via LlamaIndex's QdrantVectorStore
# Create Qdrant client using environment variables
client = qdrant_client.QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# Create vector store with the client
vector_store = QdrantVectorStore(
    collection_name="allplan_docs_collection",
    client=client,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. Build index: embeds nodes with Jina and stores in Qdrant
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

print("VectorStoreIndex built and documents stored in Qdrant")

# %%

# 6. Create retriever (uses Settings.embed_model configured above)
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

# 7. Configure LLM for answer generation using Anthropic API key
chat_model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
    max_retries=2,
    api_key=anthropic_api_key,
)

llama_llm = LangChainLLM(llm=chat_model)


# Wrap LlamaIndex components in a RAG-compatible class
class LlamaIndexRAG(RAG):
    """RAG implementation using LlamaIndex Core, Jina embeddings, and Qdrant."""

    def __init__(
        self,
        llm,
        documents: List[Document],
        k: int = 3,
        index: VectorStoreIndex = None,
        retriever: VectorIndexRetriever = None,
        query_engine: RetrieverQueryEngine = None,
    ):
        super().__init__(llm, documents, k)
        self.index = index
        self.retriever = retriever

        # Create query engine if not provided
        if query_engine is None and self.retriever is not None:
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=self.retriever,
                llm=llama_llm,
                response_mode=ResponseMode.COMPACT,
            )
        else:
            self.query_engine = query_engine

    def retrieve(self, question: str, *args: Any, **kwargs: Any) -> List[Document]:
        """Retrieve relevant documents using LlamaIndex retriever."""
        if self.retriever is None:
            return []

        # Retrieve nodes from LlamaIndex
        nodes = self.retriever.retrieve(question)

        # Convert LlamaIndex nodes to LangChain Documents (for the eval framework)
        langchain_docs = []
        for node in nodes[: self.k]:
            doc = Document(
                page_content=node.get_content(),
                metadata=getattr(node, "metadata", {}) or {},
            )
            langchain_docs.append(doc)

        return langchain_docs

    def generate(self, question: str, context: str, *args: Any, **kwargs: Any) -> str:
        """Generate answer using LlamaIndex query engine (retrieval + synthesis)."""
        if self.query_engine is None:
            # Fallback: simple LLM call using raw context
            prompt = (
                "Using the following context, answer the question:\n\n"
                f"Context: {context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
            answer = self.llm.invoke(prompt)
            if hasattr(answer, "content"):
                return answer.content
            return str(answer)

        # Preferred path: use LlamaIndex query engine
        response = self.query_engine.query(question)
        return str(response)


# Create RAG instance
rag = LlamaIndexRAG(
    llm=chat_model,
    documents=documents,
    k=3,
    index=index,
    retriever=retriever,
)

# %%

# Test the RAG pipeline
query = "What is Allplan?"

the_relevant_docs = rag.retrieve(question=query)
print(f"Retrieved {len(the_relevant_docs)} relevant documents")

answer, relevant_docs = rag.answer(question=query)
print(f"\nQuestion: {query}")
print(f"\nAnswer: {answer}")
print(f"\nRetrieved {len(relevant_docs)} documents")

# %%
# Get annotated data for evaluation
json = JSONService(path="generated_qa_data_tum.json",
                   root_path=glob.DATA_PKG_DIR, verbose=True)

qa_data = json.doRead()
print(f"Number of evaluation data samples: {len(qa_data)}")

ground_truth_contexts = [item["context"] for item in qa_data]
sample_queries = [item["question"] for item in qa_data]
expected_responses = [item["answer"] for item in qa_data]

# %%
# Create the builder with the RAG instance
builder = eval_builder.EvalDatasetBuilder(rag)

# Build the evaluation dataset
evaluation_dataset = builder.build_evaluation_dataset(
    input_contexts=ground_truth_contexts,
    sample_queries=sample_queries,
    expected_responses=expected_responses,
)

# %%

scorer = deep.DeepEvalScorer(evaluation_dataset)

results = scorer.calculate_scores()
print(results)

# %%
scorer.get_overall_metrics()

# %%
scorer.get_summary(save_to_file=True)
