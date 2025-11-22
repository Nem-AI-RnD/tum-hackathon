"""
Compare generated RAG answers with original answers in the dataset.

This script:
1. Loads the original QA dataset
2. Loads the RAG results
3. Compares generated answers with original answers
4. Calculates similarity metrics
5. Saves comparison results
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from difflib import SequenceMatcher

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON file."""
    logger.info(f"Loading {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using sequence matching."""
    if not text1 or not text2:
        return 0.0

    # Normalize texts
    text1 = text1.strip().lower()
    text2 = text2.strip().lower()

    # Use SequenceMatcher for similarity
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def compare_answers(original: Dict[str, Any], generated: Dict[str, Any]) -> Dict[str, Any]:
    """Compare original and generated answers for a single question."""
    original_answer = original.get("answer", "")
    generated_answer = generated.get("generated_answer", "")

    # Check if generation was successful
    is_error = generated_answer.startswith("ERROR:")

    # Calculate similarity
    similarity = 0.0 if is_error else calculate_similarity(
        original_answer, generated_answer)

    # Compare contexts
    original_context = original.get("context", "")
    generated_context = generated.get("generated_context", "")
    context_similarity = calculate_similarity(
        original_context, generated_context)

    # Check if contexts are identical
    contexts_match = original_context.strip() == generated_context.strip()

    return {
        "index": original.get("index", -1),
        # Truncate for readability
        "question": original.get("question", "")[:100] + "...",
        "original_answer_length": len(original_answer),
        "generated_answer_length": len(generated_answer),
        "answer_similarity": similarity,
        "context_similarity": context_similarity,
        "contexts_match": contexts_match,
        "num_retrieved_docs": generated.get("num_retrieved_docs", 0),
        "is_error": is_error,
    }


def generate_comparison_report(
    original_dataset: List[Dict[str, Any]],
    rag_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate comprehensive comparison report."""
    logger.info("Generating comparison report...")

    # Ensure both datasets have same length
    if len(original_dataset) != len(rag_results):
        logger.warning(
            f"Dataset size mismatch: original={len(original_dataset)}, "
            f"results={len(rag_results)}"
        )

    # Compare each pair
    comparisons = []
    for orig, gen in zip(original_dataset, rag_results):
        comparison = compare_answers(orig, gen)
        comparisons.append(comparison)

    # Calculate aggregate statistics
    successful = [c for c in comparisons if not c["is_error"]]
    failed = [c for c in comparisons if c["is_error"]]

    report = {
        "total_questions": len(comparisons),
        "successful_generations": len(successful),
        "failed_generations": len(failed),
        "statistics": {},
        "comparisons": comparisons,
    }

    if successful:
        similarities = [c["answer_similarity"] for c in successful]
        context_similarities = [c["context_similarity"] for c in successful]
        contexts_match_count = sum(
            1 for c in successful if c["contexts_match"])

        report["statistics"] = {
            "answer_similarity": {
                "mean": sum(similarities) / len(similarities),
                "min": min(similarities),
                "max": max(similarities),
                "median": sorted(similarities)[len(similarities) // 2],
            },
            "context_similarity": {
                "mean": sum(context_similarities) / len(context_similarities),
                "min": min(context_similarities),
                "max": max(context_similarities),
                "median": sorted(context_similarities)[len(context_similarities) // 2],
            },
            "exact_context_matches": contexts_match_count,
            "exact_context_match_rate": contexts_match_count / len(successful),
        }

        # Find best and worst matches
        report["best_matches"] = sorted(
            successful, key=lambda x: x["answer_similarity"], reverse=True)[:5]
        report["worst_matches"] = sorted(
            successful, key=lambda x: x["answer_similarity"])[:5]

    return report


def print_summary(report: Dict[str, Any]) -> None:
    """Print summary of comparison report."""
    print("\n" + "="*80)
    print("RAG ANSWER COMPARISON REPORT")
    print("="*80)

    print(f"\nTotal Questions: {report['total_questions']}")
    print(f"Successful: {report['successful_generations']}")
    print(f"Failed: {report['failed_generations']}")

    if report.get("statistics"):
        stats = report["statistics"]
        print("\n" + "-"*80)
        print("ANSWER SIMILARITY METRICS")
        print("-"*80)

        ans_sim = stats["answer_similarity"]
        print(f"Mean:   {ans_sim['mean']:.3f}")
        print(f"Median: {ans_sim['median']:.3f}")
        print(f"Min:    {ans_sim['min']:.3f}")
        print(f"Max:    {ans_sim['max']:.3f}")

        print("\n" + "-"*80)
        print("CONTEXT SIMILARITY METRICS")
        print("-"*80)

        ctx_sim = stats["context_similarity"]
        print(f"Mean:   {ctx_sim['mean']:.3f}")
        print(f"Median: {ctx_sim['median']:.3f}")
        print(f"Min:    {ctx_sim['min']:.3f}")
        print(f"Max:    {ctx_sim['max']:.3f}")

        print("\n" + "-"*80)
        print("CONTEXT MATCHING")
        print("-"*80)
        print(
            f"Exact matches: {stats['exact_context_matches']} / {report['successful_generations']}")
        print(f"Match rate: {stats['exact_context_match_rate']:.1%}")

        if report.get("best_matches"):
            print("\n" + "-"*80)
            print("TOP 5 BEST MATCHES")
            print("-"*80)
            for i, match in enumerate(report["best_matches"], 1):
                print(
                    f"\n{i}. Index: {match['index']}, Similarity: {match['answer_similarity']:.3f}")
                print(f"   Question: {match['question']}")

        if report.get("worst_matches"):
            print("\n" + "-"*80)
            print("TOP 5 WORST MATCHES")
            print("-"*80)
            for i, match in enumerate(report["worst_matches"], 1):
                print(
                    f"\n{i}. Index: {match['index']}, Similarity: {match['answer_similarity']:.3f}")
                print(f"   Question: {match['question']}")

    print("\n" + "="*80)


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent
    original_path = project_root / "data" / "generated_qa_data_tum.json"
    results_path = project_root / "data" / "rag_results.json"
    output_path = project_root / "data" / "comparison_results.json"

    try:
        # Check if results file exists
        if not results_path.exists():
            logger.error(f"Results file not found: {results_path}")
            logger.error(
                "Please run generate_simple_rag_answers.py or generate_rag_answers.py first")
            sys.exit(1)

        # Load data
        original_dataset = load_json(original_path)
        rag_results = load_json(results_path)

        # Generate comparison report
        report = generate_comparison_report(original_dataset, rag_results)

        # Save report
        logger.info(f"Saving comparison report to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Print summary
        print_summary(report)

        logger.info(f"\nDetailed comparison saved to: {output_path}")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
