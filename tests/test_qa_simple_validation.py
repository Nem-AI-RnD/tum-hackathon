"""
Simple tests for QA dataset validation using Claude Haiku.

These tests require ANTHROPIC_API_KEY to be set in a .env file in the project root and:
- Parse the QA dataset JSON file
- Use Claude Haiku (small model) to check if answers are correct
- Return test passed if confidence is high enough
- Do not rely on deepeval

The .env file should be located at: <project_root>/.env
Where <project_root> is the directory containing the tests/ folder.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import pytest  # type: ignore[import-untyped]
from dotenv import load_dotenv  # type: ignore[import-untyped]
from langchain_anthropic import ChatAnthropic  # type: ignore[import-untyped]

# Load environment variables from .env file in project root
# Project root is the parent directory of the tests/ folder
_project_root = Path(__file__).parent.parent
_env_file = _project_root / ".env"
load_dotenv(_env_file, override=False)

# Minimum confidence threshold for passing tests (0.0 to 1.0)
MIN_CONFIDENCE_THRESHOLD = 0.7
API_KEY_ERROR_MESSAGE = (
    "ANTHROPIC_API_KEY environment variable is required. "
    "Please set it in a .env file located at the project root (same directory as tests/ folder). "
    "Claude Haiku API must be available for answer validation."
)


class EvaluatorLLM(Protocol):
    """Protocol for objects that expose an Anthropic-like invoke method."""

    def invoke(self, prompt: str) -> Any:  # pragma: no cover - typing helper
        ...


def load_qa_dataset(file_path: Path) -> List[Dict]:
    """Load QA dataset from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must contain a list of items")
    return data


def validate_answer_with_claude(
    question: str,
    answer: str,
    context: str,
    target_answer: Optional[str] = None,
    llm: Optional[EvaluatorLLM] = None,
) -> Dict[str, float]:
    """
    Use Claude Haiku to validate if an answer is correct by comparing with target answer.

    Args:
        question: The question
        answer: The actual answer to validate
        context: The context/reference material
        target_answer: The expected/target answer
        llm: Optional ChatAnthropic instance

    Returns:
        Dictionary with 'confidence', 'is_correct', 'semantic_match', and 'reasoning'
    """
    if llm is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(API_KEY_ERROR_MESSAGE)
        try:
            llm = ChatAnthropic(
                model="claude-haiku-4-5-20251001",  # Claude Haiku
                temperature=0.0,
                max_tokens=800,
            )
        except Exception as exc:  # pragma: no cover - dependency issues
            raise RuntimeError(API_KEY_ERROR_MESSAGE) from exc

    # Create evaluation prompt comparing actual with target answer
    prompt = f"""You are an expert evaluator comparing an actual answer with a target answer.

Question: {question}

Context: {context}

Actual Answer: {answer}

Target Answer: {target_answer if target_answer else "No target provided"}

Task: Evaluate if the actual answer is semantically equivalent to the target answer and correctly addresses the question.

Consider:
- Semantic equivalence (same meaning, different wording OK)
- Factual accuracy based on context
- Completeness

Respond ONLY with valid JSON in this format:
{{
  "confidence": 0.9,
  "is_correct": true,
  "semantic_match": 0.95,
  "reasoning": "Brief explanation"
}}

Where:
- confidence: 0.0-1.0 (confidence the answer is correct)
- is_correct: true/false
- semantic_match: 0.0-1.0 (how well actual matches target)
- reasoning: brief explanation (max 2 sentences)

JSON:"""

    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(
            response, "content") else str(response)

        # Extract JSON from response
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        response_text = response_text.strip()

        result = json.loads(response_text)

        return {
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            "is_correct": bool(result.get("is_correct", False)),
            "semantic_match": max(0.0, min(1.0, float(result.get("semantic_match", 0.0)))),
            "reasoning": result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return {
            "confidence": 0.3,
            "is_correct": False,
            "semantic_match": 0.0,
            "reasoning": f"Failed to parse response: {str(e)}",
        }


@pytest.fixture
def qa_dataset_path() -> Path:
    """Path to the QA dataset file."""
    return Path("data/generated_qa_data_tum.json")


@pytest.fixture
def claude_llm() -> EvaluatorLLM:
    """Create Claude Haiku LLM instance; requires Anthropic API access."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(API_KEY_ERROR_MESSAGE)

    try:
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",  # Claude Haiku
            temperature=0.0,
            max_tokens=800,
        )
    except Exception as exc:  # pragma: no cover - dependency issues
        raise RuntimeError(API_KEY_ERROR_MESSAGE) from exc


class TestQASimpleValidation:
    """Simple tests for QA dataset validation using Claude Haiku."""

    def test_load_qa_dataset(self, qa_dataset_path: Path):
        """Test that the QA dataset can be loaded."""
        assert qa_dataset_path.exists(
        ), f"Dataset file not found: {qa_dataset_path}"

        dataset = load_qa_dataset(qa_dataset_path)
        assert len(dataset) > 0
        assert isinstance(dataset, list)

        # Check structure of first item
        first_item = dataset[0]
        assert "question" in first_item
        assert "answer" in first_item
        assert "context" in first_item

    def test_validate_sample_answers(
        self, qa_dataset_path: Path, claude_llm: EvaluatorLLM
    ):
        """Test validating a sample of answers from the dataset."""
        assert qa_dataset_path.exists(
        ), f"Dataset file not found: {qa_dataset_path}"

        dataset = load_qa_dataset(qa_dataset_path)

        # Test first 5 items (or all if less than 5)
        sample_size = min(5, len(dataset))
        test_items = dataset[:sample_size]

        passed = 0
        total_confidence = 0.0

        for item in test_items:
            question = item.get("question", "")
            answer = item.get("answer", "")
            context = item.get("context", "")
            target_answer = item.get(
                "location_dependency_evaluator_target_answer")

            if not all([question, answer, context]):
                continue

            result = validate_answer_with_claude(
                question=question,
                answer=answer,
                context=context,
                target_answer=target_answer,
                llm=claude_llm,
            )

            confidence = result["confidence"]
            is_correct = result["is_correct"]
            semantic_match = result.get("semantic_match", 0.0)
            total_confidence += confidence

            # Test passes if confidence is above threshold
            if confidence >= MIN_CONFIDENCE_THRESHOLD:
                passed += 1

            print(f"\nQuestion {item.get('index', '?')}:")
            print(f"  Answer: {answer[:80]}...")
            print(
                f"  Target: {target_answer[:80] if target_answer else 'N/A'}...")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Semantic Match: {semantic_match:.2f}")
            print(f"  Is Correct: {is_correct}")
            print(f"  Reasoning: {result.get('reasoning', '')[:150]}")

        avg_confidence = total_confidence / \
            len(test_items) if test_items else 0.0

        print("\nSummary:")
        print(f"  Tested: {len(test_items)} items")
        print(
            f"  Passed (confidence >= {MIN_CONFIDENCE_THRESHOLD}): {passed}/{len(test_items)}")
        print(f"  Average confidence: {avg_confidence:.2f}")

        # Test passes only if at least 70% of items have high confidence
        pass_rate = passed / len(test_items) if test_items else 0.0
        assert pass_rate >= 0.7, (
            f"Only {passed}/{len(test_items)} items passed "
            f"(expected at least 70%). Average confidence: {avg_confidence:.2f}"
        )
        assert avg_confidence >= MIN_CONFIDENCE_THRESHOLD, (
            f"Average confidence {avg_confidence:.2f} below threshold "
            f"{MIN_CONFIDENCE_THRESHOLD}"
        )

    def test_validate_all_high_quality_answers(
        self, qa_dataset_path: Path, claude_llm: EvaluatorLLM
    ):
        """Test validating all high-quality answers (scores >= 4)."""
        assert qa_dataset_path.exists(
        ), f"Dataset file not found: {qa_dataset_path}"

        dataset = load_qa_dataset(qa_dataset_path)

        # Filter for high-quality items (all scores >= 4)
        high_quality_items = [
            item for item in dataset
            if all([
                item.get("groundedness_score", 0) >= 4,
                item.get("question_relevancy_score", 0) >= 4,
                item.get("faithfulness_score", 0) >= 4,
            ])
        ]

        assert len(
            high_quality_items) > 0, "No high-quality items found in dataset"

        # Test first 10 high-quality items
        sample_size = min(10, len(high_quality_items))
        test_items = high_quality_items[:sample_size]

        passed = 0
        total_confidence = 0.0

        for item in test_items:
            question = item.get("question", "")
            answer = item.get("answer", "")
            context = item.get("context", "")
            target_answer = item.get(
                "location_dependency_evaluator_target_answer")

            if not all([question, answer, context]):
                continue

            result = validate_answer_with_claude(
                question=question,
                answer=answer,
                context=context,
                target_answer=target_answer,
                llm=claude_llm,
            )

            confidence = result["confidence"]
            semantic_match = result.get("semantic_match", 0.0)
            total_confidence += confidence

            if confidence >= MIN_CONFIDENCE_THRESHOLD:
                passed += 1

            print(
                f"  Item {item.get('index', '?')}: confidence={confidence:.2f}, semantic_match={semantic_match:.2f}")

        avg_confidence = total_confidence / \
            len(test_items) if test_items else 0.0

        print("\nHigh-Quality Items Validation:")
        print(f"  Tested: {len(test_items)} items")
        print(
            f"  Passed (confidence >= {MIN_CONFIDENCE_THRESHOLD}): {passed}/{len(test_items)}")
        print(f"  Average confidence: {avg_confidence:.2f}")

        # For high-quality items, expect at least 70% to pass
        pass_rate = passed / len(test_items) if test_items else 0.0
        assert pass_rate >= 0.7, (
            f"Only {passed}/{len(test_items)} high-quality items passed "
            f"(expected at least 70%). Average confidence: {avg_confidence:.2f}"
        )

    def test_validate_single_answer(
        self, qa_dataset_path: Path, claude_llm: EvaluatorLLM
    ):
        """Test validating a single answer (useful for debugging)."""
        assert qa_dataset_path.exists(
        ), f"Dataset file not found: {qa_dataset_path}"

        dataset = load_qa_dataset(qa_dataset_path)

        assert len(dataset) > 0, "Dataset is empty"

        # Test first item
        item = dataset[0]
        question = item.get("question", "")
        answer = item.get("answer", "")
        context = item.get("context", "")
        expected_answer = item.get(
            "location_dependency_evaluator_target_answer")

        result = validate_answer_with_claude(
            question=question,
            answer=answer,
            context=context,
            target_answer=expected_answer,
            llm=claude_llm,
        )

        confidence = result["confidence"]
        is_correct = result["is_correct"]

        print("\nSingle Answer Validation:")
        print(f"  Question: {question[:100]}...")
        print(f"  Answer: {answer[:100]}...")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Is Correct: {is_correct}")
        print(f"  Reasoning: {result.get('reasoning', '')}")

        # Test passes if confidence is above threshold
        assert confidence >= MIN_CONFIDENCE_THRESHOLD, (
            f"Answer validation failed with confidence {confidence:.2f} "
            f"(threshold: {MIN_CONFIDENCE_THRESHOLD})"
        )

    def test_confidence_gate_with_stub_model(self):
        """Ensure the confidence gate enforces the >70% requirement using a stub LLM."""

        class StubLLM:
            def __init__(self, payload: Dict[str, float | bool | str]):
                self.payload = payload

            def invoke(self, prompt: str):  # noqa: ARG002 - prompt logged implicitly
                return type("Resp", (), {"content": json.dumps(self.payload)})

        high_conf_stub = StubLLM(
            {
                "confidence": 0.72,
                "is_correct": True,
                "semantic_match": 0.81,
                "reasoning": "Answers align closely.",
            }
        )
        low_conf_stub = StubLLM(
            {
                "confidence": 0.49,
                "is_correct": False,
                "semantic_match": 0.3,
                "reasoning": "Insufficient overlap.",
            }
        )

        high_result = validate_answer_with_claude(
            question="What governs fire safety in Section 5.2?",
            answer="Section 5.2 requires reinforced concrete walls.",
            context="Section 5.2 of the handbook states reinforced walls are mandatory.",
            target_answer="Reinforced walls per Section 5.2.",
            llm=high_conf_stub,  # type: ignore[arg-type]
        )
        low_result = validate_answer_with_claude(
            question="What governs fire safety in Section 5.2?",
            answer="Use wooden frames.",
            context="Section 5.2 of the handbook states reinforced walls are mandatory.",
            target_answer="Reinforced walls per Section 5.2.",
            llm=low_conf_stub,  # type: ignore[arg-type]
        )

        assert high_result["confidence"] >= MIN_CONFIDENCE_THRESHOLD
        assert low_result["confidence"] < MIN_CONFIDENCE_THRESHOLD
