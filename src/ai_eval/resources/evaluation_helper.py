"""
Evaluation Helper for setup.ipynb Integration

This module provides convenience functions to integrate RAGPrototype
with the setup.ipynb evaluation workflow and DeepEval scoring.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase

from ai_eval.resources.rag_prototype import RAGPrototype, QADataItem

logger = logging.getLogger(__name__)


class EvaluationHelper:
    """
    Helper class to bridge RAGPrototype with DeepEval evaluation workflow.
    
    This class provides convenient methods for the setup.ipynb notebook
    to seamlessly integrate RAGPrototype with DeepEvalScorer.
    """

    def __init__(self, rag: RAGPrototype):
        """
        Initialize evaluation helper.
        
        Args:
            rag: RAGPrototype instance to evaluate
        """
        self.rag = rag

    def create_evaluation_dataset(
        self,
        questions: List[str],
        expected_answers: Optional[List[str]] = None,
        ground_truth_contexts: Optional[List[str]] = None,
        run_rag: bool = True,
        show_progress: bool = True,
    ) -> EvaluationDataset:
        """
        Create DeepEval EvaluationDataset from questions.
        
        This is the main method for setup.ipynb integration, replacing
        the manual EvalDatasetBuilder workflow.
        
        Args:
            questions: List of questions to evaluate
            expected_answers: Optional ground truth answers
            ground_truth_contexts: Optional ground truth contexts
            run_rag: Whether to run RAG pipeline (True) or use expected values (False)
            show_progress: Show progress during evaluation
            
        Returns:
            EvaluationDataset ready for DeepEvalScorer
            
        Example:
            ```python
            # In setup.ipynb:
            from ai_eval.resources.evaluation_helper import EvaluationHelper
            
            helper = EvaluationHelper(rag_prototype)
            evaluation_dataset = helper.create_evaluation_dataset(
                questions=sample_queries,
                expected_answers=expected_responses,
                ground_truth_contexts=ground_truth_contexts,
                run_rag=True
            )
            
            # Now use with DeepEvalScorer
            scorer = DeepEvalScorer(evaluation_dataset)
            results = scorer.calculate_scores()
            ```
        """
        logger.info("📊 Creating EvaluationDataset for %d questions...", len(questions))
        
        goldens = []
        test_cases = []
        
        total = len(questions)
        for idx, question in enumerate(questions):
            if show_progress and (idx + 1) % 10 == 0:
                logger.info("   Progress: %d/%d", idx + 1, total)
            
            try:
                # Get expected values
                expected_answer = expected_answers[idx] if expected_answers else ""
                gt_context = ground_truth_contexts[idx] if ground_truth_contexts else ""
                
                # Create golden (ground truth)
                golden = Golden(
                    input=question,
                    expected_output=expected_answer,
                    context=[gt_context] if gt_context else []
                )
                goldens.append(golden)
                
                # Get actual output and retrieval context
                if run_rag:
                    answer, relevant_docs = self.rag.answer(question)
                    retrieval_contexts = [doc.page_content for doc in relevant_docs]
                else:
                    answer = expected_answer
                    retrieval_contexts = [gt_context] if gt_context else []
                
                # Create test case
                test_case = LLMTestCase(
                    input=question,
                    expected_output=expected_answer,
                    actual_output=answer,
                    context=[gt_context] if gt_context else [],
                    retrieval_context=retrieval_contexts,
                )
                test_cases.append(test_case)
                
            except Exception as e:
                logger.error("❌ Error processing question %d: %s", idx, e)
                # Create empty test case to maintain index alignment
                golden = Golden(input=question, expected_output="", context=[])
                goldens.append(golden)
                
                test_case = LLMTestCase(
                    input=question,
                    expected_output="",
                    actual_output="",
                    context=[],
                    retrieval_context=[],
                )
                test_cases.append(test_case)
                continue
        
        # Create dataset
        dataset = EvaluationDataset(goldens=goldens, test_cases=test_cases)
        
        logger.info("✅ EvaluationDataset created with %d test cases", len(test_cases))
        return dataset

    def create_evaluation_dataset_from_qa_data(
        self,
        qa_data: List[Dict[str, Any]],
        run_rag: bool = False,
    ) -> EvaluationDataset:
        """
        Create EvaluationDataset from existing QA data (generated_qa_data_tum.json format).
        
        Args:
            qa_data: List of QA data dictionaries
            run_rag: Whether to re-run RAG pipeline (False = use existing answers)
            
        Returns:
            EvaluationDataset ready for DeepEvalScorer
            
        Example:
            ```python
            # Load existing QA data
            qa_data = json.load(open("data/generated_qa_data_tum.json"))
            
            helper = EvaluationHelper(rag_prototype)
            evaluation_dataset = helper.create_evaluation_dataset_from_qa_data(
                qa_data=qa_data,
                run_rag=False  # Use existing answers
            )
            ```
        """
        questions = [item["question"] for item in qa_data]
        expected_answers = [item["answer"] for item in qa_data]
        contexts = [item["context"] for item in qa_data]
        
        return self.create_evaluation_dataset(
            questions=questions,
            expected_answers=expected_answers,
            ground_truth_contexts=contexts,
            run_rag=run_rag,
        )

    def evaluate_and_compare(
        self,
        questions: List[str],
        expected_answers: Optional[List[str]] = None,
        ground_truth_contexts: Optional[List[str]] = None,
        baseline_path: Optional[str] = None,
        save_results: bool = True,
        output_dir: str = "data",
    ) -> Dict[str, Any]:
        """
        Complete evaluation workflow with optional baseline comparison.
        
        This is a convenience method that combines dataset creation,
        evaluation, and optional baseline comparison in one call.
        
        Args:
            questions: Questions to evaluate
            expected_answers: Expected answers
            ground_truth_contexts: Ground truth contexts
            baseline_path: Optional path to baseline results for comparison
            save_results: Save results to files
            output_dir: Directory for output files
            
        Returns:
            Dictionary with evaluation results and optional comparison
            
        Example:
            ```python
            helper = EvaluationHelper(rag_prototype)
            results = helper.evaluate_and_compare(
                questions=sample_queries,
                expected_answers=expected_responses,
                ground_truth_contexts=ground_truth_contexts,
                baseline_path="data/baseline_results.json",
                save_results=True
            )
            ```
        """
        # Run evaluation
        results = self.rag.run_deepeval_evaluation(
            questions=questions,
            expected_answers=expected_answers,
            ground_truth_contexts=ground_truth_contexts,
            save_results=save_results,
            output_dir=output_dir,
        )
        
        # Compare with baseline if provided
        if baseline_path:
            comparison = self.rag.compare_with_baseline(
                baseline_results_path=baseline_path,
                current_results=results,
            )
            results["baseline_comparison"] = comparison
            
            if save_results:
                comparison_path = Path(output_dir) / "comparison_results.json"
                import json
                with open(comparison_path, "w", encoding="utf-8") as f:
                    json.dump(comparison, f, indent=2, ensure_ascii=False)
                logger.info("💾 Comparison saved to: %s", comparison_path)
        
        return results

    def batch_evaluate_multiple_configs(
        self,
        questions: List[str],
        expected_answers: List[str],
        ground_truth_contexts: List[str],
        configs: List[Dict[str, Any]],
        output_dir: str = "data/batch_evaluation",
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple RAG configurations in batch.
        
        Args:
            questions: Questions to evaluate
            expected_answers: Expected answers
            ground_truth_contexts: Ground truth contexts
            configs: List of configuration overrides to test
            output_dir: Directory for outputs
            
        Returns:
            List of results for each configuration
            
        Example:
            ```python
            configs = [
                {"chunk_size": 512, "top_k": 3},
                {"chunk_size": 1024, "top_k": 5},
                {"chunk_size": 2048, "top_k": 7},
            ]
            
            helper = EvaluationHelper(rag_prototype)
            batch_results = helper.batch_evaluate_multiple_configs(
                questions=sample_queries,
                expected_answers=expected_responses,
                ground_truth_contexts=ground_truth_contexts,
                configs=configs
            )
            ```
        """
        from ai_eval.resources.rag_prototype import RAGConfig
        
        all_results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔄 Starting batch evaluation of %d configurations...", len(configs))
        
        for idx, config_override in enumerate(configs):
            logger.info("\n" + "="*60)
            logger.info("Evaluating configuration %d/%d", idx + 1, len(configs))
            logger.info("="*60)
            
            try:
                # Create new config with overrides
                base_config = self.rag.config.model_dump()
                base_config.update(config_override)
                base_config["collection_name"] = f"batch_eval_{idx}"
                
                new_config = RAGConfig(**base_config)
                
                # Create new RAG instance
                from ai_eval.resources.rag_prototype import RAGPrototype
                rag = RAGPrototype(config=new_config)
                
                # Deploy embeddings
                rag.deploy_embeddings(
                    self.rag.documents if hasattr(self.rag, 'documents') else [],
                    force_redeploy=True
                )
                
                # Run evaluation
                helper = EvaluationHelper(rag)
                results = helper.evaluate_and_compare(
                    questions=questions,
                    expected_answers=expected_answers,
                    ground_truth_contexts=ground_truth_contexts,
                    save_results=True,
                    output_dir=str(output_path / f"config_{idx}")
                )
                
                results["config_overrides"] = config_override
                results["config_index"] = idx
                all_results.append(results)
                
            except Exception as e:
                logger.error("❌ Configuration %d failed: %s", idx, e)
                all_results.append({
                    "config_index": idx,
                    "config_overrides": config_override,
                    "error": str(e)
                })
        
        # Save batch summary
        import json
        summary_path = output_path / "batch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info("\n✅ Batch evaluation complete!")
        logger.info("   Evaluated: %d configurations", len(configs))
        logger.info("   Results saved to: %s", output_path)
        
        return all_results


def create_evaluation_dataset_quick(
    rag: RAGPrototype,
    questions: List[str],
    expected_answers: Optional[List[str]] = None,
    ground_truth_contexts: Optional[List[str]] = None,
) -> EvaluationDataset:
    """
    Quick helper function to create EvaluationDataset.
    
    This is a convenience function for use in setup.ipynb notebooks.
    
    Args:
        rag: RAGPrototype instance
        questions: Questions to evaluate
        expected_answers: Expected answers
        ground_truth_contexts: Ground truth contexts
        
    Returns:
        EvaluationDataset ready for DeepEvalScorer
        
    Example:
        ```python
        # In setup.ipynb - one-liner:
        from ai_eval.resources.evaluation_helper import create_evaluation_dataset_quick
        
        evaluation_dataset = create_evaluation_dataset_quick(
            rag_prototype,
            sample_queries,
            expected_responses,
            ground_truth_contexts
        )
        
        # Use with DeepEvalScorer
        scorer = DeepEvalScorer(evaluation_dataset)
        results = scorer.calculate_scores()
        ```
    """
    helper = EvaluationHelper(rag)
    return helper.create_evaluation_dataset(
        questions=questions,
        expected_answers=expected_answers,
        ground_truth_contexts=ground_truth_contexts,
        run_rag=True,
    )


__all__ = [
    "EvaluationHelper",
    "create_evaluation_dataset_quick",
]

