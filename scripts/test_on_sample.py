"""
Quick test script to generate RAG answers for a small sample of questions.

This is useful for testing the pipeline without processing all 100 questions.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_anthropic import ChatAnthropic
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_eval.resources.llm_aaj import answer_with_rag_tfidf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def main():
    """Test RAG on a small sample."""
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "generated_qa_data_tum.json"
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Take only first 5 questions for testing
    sample = dataset[:5]
    logger.info(f"Testing on {len(sample)} questions")
    
    # Extract documents
    unique_contexts = {}
    for item in dataset:
        context = item.get("context", "").strip()
        if context and context not in unique_contexts:
            unique_contexts[context] = Document(
                page_content=context,
                metadata={"index": item.get("index", -1)}
            )
    documents = list(unique_contexts.values())
    logger.info(f"Loaded {len(documents)} unique documents")
    
    # Initialize LLM
    logger.info("Initializing LLM...")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment")
        sys.exit(1)
    
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.0,
        max_tokens=1024,
    )
    
    # Generate answers
    print("\n" + "="*80)
    print("GENERATING SAMPLE RAG ANSWERS")
    print("="*80 + "\n")
    
    for item in tqdm(sample, desc="Processing"):
        question = item["question"]
        original_answer = item["answer"]
        
        logger.info(f"\nQuestion {item['index']}: {question[:80]}...")
        
        try:
            generated_answer, relevant_docs = answer_with_rag_tfidf(
                question=question,
                llm=llm,
                documents=documents,
                k=3
            )
            
            print(f"\nQuestion {item['index']}:")
            print(f"Q: {question}")
            print(f"\nOriginal Answer:")
            print(f"  {original_answer}")
            print(f"\nGenerated Answer:")
            print(f"  {generated_answer}")
            print(f"\nRetrieved {len(relevant_docs)} documents")
            print("-" * 80)
            
        except Exception as e:
            logger.error(f"Error: {e}")
    
    print("\n" + "="*80)
    print("Sample test complete!")
    print("To process all questions, run: python scripts/generate_simple_rag_answers.py")
    print("="*80)


if __name__ == "__main__":
    main()

