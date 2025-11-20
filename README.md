# TUM Hackathon - RAG Evaluation Framework

This repository contains the code utilities and examples developed for the **TUM Hackathon**, focused on building and evaluating Retrieval-Augmented Generation (RAG) systems. It provides a complete pipeline for:

- 🔄 **Synthetic test data generation** from PDF documents
- 🏗️ **Custom RAG model implementation** with flexible retrieval strategies
- 📊 **Automated evaluation** using LLM-as-judge metrics (DeepEval)
- 🚀 **Production-ready components** powered by Google Cloud Vertex AI and LangChain

## Package structure

```
├── .env                          # Environment configuration (GCP credentials)
├── pyproject.toml                # Python package configuration
├── requirements.txt              # Pip-compatible dependencies
├── uv.lock                       # UV dependency lock file
├── data/                         # Document corpus and generated datasets
│   ├── Allplan_2020_Manual.pdf
│   ├── generated_qa_data_tum.json
├── src/
│   ├── ai_eval/                  # Core evaluation framework
│   │   ├── config/               # Configuration management
│   │   │   ├── config.py
│   │   │   ├── global_config.py  # Environment variables & settings
│   │   │   ├── input_output.yaml
│   │   │   └── model_config.yaml
│   │   ├── resources/            # RAG and evaluation components
│   │   │   ├── synthesizer.py    # QA data generation
│   │   │   ├── rag_template.py   # RAG model templates (TF-IDF, etc.)
│   │   │   ├── deepeval_scorer.py # Evaluation metrics
│   │   │   ├── eval_dataset_builder.py
│   │   │   ├── get_models.py     # Vertex AI model initialization
│   │   │   ├── preprocessor.py   # Document chunking
│   │   │   ├── prompts.py        # Prompt templates
│   │   │   ├── llm_aaj.py        # LLM utilities
│   │   │   └── data_schemas.py
│   │   ├── services/             # Infrastructure services
│   │   │   ├── logger.py         # Centralized logging
│   │   │   ├── database.py       # Firestore integration
│   │   │   ├── file.py           # File operations
│   │   │   └── blueprint_file.py
│   │   └── utils/                # Helper utilities
│   │       └── utils.py
│   └── notebooks/                # Example workflows
│       └── prepare_tum_hackathon.ipynb  # End-to-end RAG evaluation demo
└── tests/                        # Unit tests
```

## Package installation and application development

### Option 1: Using `uv` (Recommended - Fast & Modern)

[`uv`](https://docs.astral.sh/uv/) is an extremely fast Python package installer and resolver, written in Rust. It's recommended for its speed and reliability.

**Step 1: Install `uv`**

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

**Step 2: Create and activate virtual environment**

```bash
# Create a Python 3.12 virtual environment
uv venv --python 3.12

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

**Step 3: Install dependencies**

```bash
# Install all dependencies from pyproject.toml
uv sync

# Or install with development dependencies
uv sync --all-extras
```

**Step 4: Install the package in editable mode** (optional)

```bash
uv pip install -e .
```

### Option 2: Using `pip` (Traditional Approach)

If you prefer using traditional `pip`, follow these steps:

**Step 1: Create virtual environment**

```bash
# Create virtual environment with Python 3.12
python3.12 -m venv .venv

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

**Step 2: Upgrade pip**

```bash
pip install --upgrade pip
```

**Step 3: Install dependencies**

```bash
# Option A: Install from pyproject.toml
pip install -e .

# Option B: Install from requirements.txt
pip install -r requirements.txt

# Option C: Install with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check installed packages
uv pip list  # if using uv
# or
pip list     # if using pip

# Test imports
python -c "import ai_eval; print('Installation successful!')"
```

### Installing Development Dependencies

For contributors and developers:

```bash
# Using uv
uv pip install ".[dev]"

# Using pip
pip install -e ".[dev]"
```

This installs additional tools like `jupyter`, `pytest`, and `pytest-timeout` for testing and development.


## Building and Evaluating Custom RAG Models

After generating a ground truth evaluation dataset (e.g., `generated_qa_data_tum.json`), you can build and evaluate custom RAG models using the evaluation pipeline.

### Technology Stack

This project leverages the following key technologies:

- **[Google Cloud Vertex AI](https://cloud.google.com/vertex-ai)**: Provides LLM services for question generation and answering (using `gemini-2.0-flash` model)
- **[LangChain](https://python.langchain.com/)**: Framework for chaining retrieval and generation components in the RAG pipeline
- **[DeepEval](https://docs.confident-ai.com/)**: LLM-as-judge evaluation framework for measuring RAG quality with metrics like Answer Relevancy, Faithfulness, Contextual Recall, and Contextual Precision

### RAG Model Architecture

A RAG (Retrieval-Augmented Generation) model consists of two main components:

1. **Retriever**: Searches through document chunks to find relevant context for a given query
2. **Generator**: Uses the retrieved context (via LangChain) and Vertex AI LLMs to generate an answer to the query

### Implementation Steps

#### 1. Create a Custom RAG Model

Follow the base template pattern in `src/ai_eval/resources/rag_template.py`. Here are examples using the provided implementations:

**Example 1: TF-IDF Retrieval (Simple & Fast)**

```python
from ai_eval.resources.rag_template import TFIDFRAG
from langchain.docstore.document import Document

# Initialize your LLM (examples for different providers)
# Option 1: OpenAI
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4", temperature=0)

# Option 2: Anthropic
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# Option 3: Ollama (local models)
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.2", temperature=0)

# Option 4: HuggingFace
# from langchain_huggingface import HuggingFaceEndpoint
# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.2-3B-Instruct",
#     task="text-generation",
#     temperature=0
# )

# Option 5: Google Vertex AI (used in this project)
# from langchain_google_vertexai import ChatVertexAI
# llm = ChatVertexAI(model="gemini-2.0-flash", temperature=0)

# Prepare your documents as LangChain Document objects
documents = [
    Document(page_content="Your document text here..."),
    Document(page_content="Another document..."),
    # ... more documents
]

# Create TF-IDF RAG instance
rag = TFIDFRAG(
    llm=llm,
    documents=documents,
    k=3  # Number of documents to retrieve
)

# Use the RAG system
answer, retrieved_docs = rag.answer("What is...?")
print(f"Answer: {answer}")
print(f"Retrieved {len(retrieved_docs)} documents")
```

**Example 2: FAISS Vector Retrieval (Semantic Search)**

```python
from ai_eval.resources.rag_template import FAISSRAG
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# Initialize your embedding model
# Option 1: HuggingFace embeddings (local or API)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Option 2: Ollama embeddings (local)
# from langchain_ollama import OllamaEmbeddings
# embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Option 3: OpenAI embeddings
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Option 4: Google Vertex AI embeddings (used in this project)
# from langchain_google_vertexai import VertexAIEmbeddings
# embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

# Initialize your chat model
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.2", temperature=0)

from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0,
    max_new_tokens=512
)

# Prepare documents
documents = [
    Document(page_content="Your document text here..."),
    Document(page_content="Another document..."),
]

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(documents, embeddings)

# Create FAISS RAG instance
rag = FAISSRAG(
    llm=llm,
    documents=documents,
    k=3,
    vectorstore=vectorstore
)

# Use the RAG system
answer, retrieved_docs = rag.answer("What is...?")
```

**Key Components:**

- **Retriever**: Choose between TF-IDF (keyword-based) or FAISS (semantic embeddings)
  - `TFIDFRAG`: Fast, no embeddings needed, good for keyword matching
  - `FAISSRAG`: Semantic search using vector embeddings, better context understanding
  
- **Generator**: Any LangChain-compatible LLM (OpenAI, Anthropic, Vertex AI, local models)
  - The LLM generates answers based on retrieved context
  - Both implementations use LangChain's prompt templates for consistency

#### 2. Build Evaluation Dataset

Use the ground truth data to create test cases with [LangChain](https://python.langchain.com/):

```python
from ai_eval.resources import eval_dataset_builder as eval

# Load ground truth data
ground_truth_contexts = [item["context"] for item in qa_data]
sample_queries = [item["question"] for item in qa_data]
expected_responses = [item["answer"] for item in qa_data]

# Create evaluation builder
builder = eval.EvalDatasetBuilder(rag)

# Build dataset - this will run your RAG model on all queries
evaluation_dataset = builder.build_evaluation_dataset(
    input_contexts=ground_truth_contexts,
    sample_queries=sample_queries,
    expected_responses=expected_responses,
)
```

#### 3. Evaluate RAG Performance

The evaluation pipeline uses [DeepEval](https://docs.confident-ai.com/) to measure both retrieval and generation quality:

```python
from ai_eval.resources import deepeval_scorer as deep

# Initialize scorer
scorer = deep.DeepEvalScorer(evaluation_dataset)

# Calculate metrics using DeepEval's LLM-as-judge approach
results = scorer.calculate_scores()

# View overall performance
metrics = scorer.get_overall_metrics()
print(f"Answer Relevancy: {metrics['Answer Relevancy']}")
print(f"Faithfulness: {metrics['Faithfulness']}")
print(f"Contextual Recall: {metrics['Contextual Recall']}")
print(f"Contextual Precision: {metrics['Contextual Precision']}")

# Save detailed results
summary = scorer.get_summary(save_to_file=True)
```

**Evaluation Metrics** (powered by DeepEval):

- **Answer Relevancy**: Does the generated answer address the query?
- **Faithfulness**: Is the answer grounded in the retrieved context (no hallucinations)?
- **Contextual Recall**: Does the retriever find all relevant information?
- **Contextual Precision**: Are the retrieved documents ranked correctly by relevance?

#### 4. Customize Your RAG Model

To implement your own retrieval strategy, extend the base RAG template:

```python
class CustomRAG:
    def __init__(self, llm, documents, k=3):
        self.llm = llm
        self.documents = documents
        self.k = k
        # Initialize your retriever (embeddings, BM25, hybrid, etc.)
        self._setup_retriever()
    
    def _setup_retriever(self):
        # Implement your retrieval logic
        # Examples: FAISS, Chroma, Pinecone, Elastic Search
        pass
    
    def retrieve(self, query: str) -> List[str]:
        # Retrieve top-k relevant documents
        # Return list of document texts
        pass
    
    def generate(self, query: str, context: List[str]) -> str:
        # Use LLM to generate answer from context
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        return self.llm.invoke(prompt)
    
    def query(self, question: str) -> tuple[str, List[str]]:
        # Full RAG pipeline
        retrieved_docs = self.retrieve(question)
        answer = self.generate(question, retrieved_docs)
        return answer, retrieved_docs
```

### Example Workflow

1. Read ground truth as JSON (`generated_qa_data_tum.json`)
2. Load documents and create document chunks
3. Initialize custom RAG model (TF-IDF example)
4. Build evaluation dataset (runs RAG on all test queries)
5. Calculate evaluation metrics
6. Analyze results and iterate on RAG design



