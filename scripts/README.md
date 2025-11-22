# RAG Answer Generation Scripts

This directory contains scripts to generate answers for the QA dataset using different RAG approaches.

## Scripts

### 1. `generate_simple_rag_answers.py` (Recommended for quick testing)

Lightweight script using TFIDF retrieval and Claude Haiku.

**Requirements:**

- `ANTHROPIC_API_KEY` in `.env` file

**Usage:**

```bash
cd /Users/karl/Documents/tum-hackathon
python scripts/generate_simple_rag_answers.py
```

**Features:**

- Fast execution (no external vector database needed)
- TFIDF-based document retrieval
- Claude Haiku for answer generation
- Retrieves top 3 most relevant documents per question

### 2. `generate_rag_answers.py` (Full-featured)

Production-grade RAG with Jina embeddings and LlamaIndex.

**Requirements:**

- `ANTHROPIC_API_KEY` in `.env` file
- `JINA_API_KEY` in `.env` file
- Qdrant (optional, can use in-memory vectorstore)

**Usage:**

```bash
cd /Users/karl/Documents/tum-hackathon
python scripts/generate_rag_answers.py
```

**Features:**

- Advanced Jina v3 embeddings
- LlamaIndex query engine
- Configurable retrieval settings
- Optional Qdrant vector database

### 3. `compare_results.py`

Compares generated answers with original answers in the dataset.

**Usage:**

```bash
python scripts/compare_results.py
```

**Output:**

- Similarity metrics between generated and original answers
- Per-question comparison statistics
- Summary report saved to `data/comparison_results.json`

## Output Format

All scripts generate results in `data/rag_results.json` with the following structure:

```json
[
  {
    "index": 0,
    "question": "Question text",
    "answer": "Original answer",
    "context": "Original context",
    "generated_answer": "RAG-generated answer",
    "generated_context": "Retrieved context used for generation",
    "num_retrieved_docs": 3,
    ...other original fields...
  }
]
```

## Quick Start

1. Ensure you have the required API keys in `.env`:

   ```
   ANTHROPIC_API_KEY=your_key_here
   # Optional for advanced RAG:
   JINA_API_KEY=your_key_here
   ```

2. Run the simple version:

   ```bash
   python scripts/generate_simple_rag_answers.py
   ```

3. Compare results:
   ```bash
   python scripts/compare_results.py
   ```

## Performance Notes

- **Simple TFIDF RAG**: ~2-5 seconds per question
- **Advanced RAG**: ~3-8 seconds per question (with embedding initialization)
- For 100 questions: expect 5-15 minutes total runtime

## Troubleshooting

**Error: "ANTHROPIC_API_KEY not found"**

- Ensure `.env` file exists in project root
- Verify the API key is correctly set

**Error: "JINA_API_KEY not found"** (only for `generate_rag_answers.py`)

- Get a free API key from https://jina.ai
- Add it to your `.env` file

**Slow execution**

- Use `generate_simple_rag_answers.py` instead
- Reduce the number of questions to process (edit the script)
- Adjust `k` parameter to retrieve fewer documents
