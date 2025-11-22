"""
Integration-style tests that verify the RAGPrototype pipeline stores content in
the vector store, retrieves it back, and produces QA data saved in the
`data/generated_qa_data_tum.json` format – without relying on Deepeval.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pytest  # type: ignore[import-untyped]
# type: ignore[import-untyped]
from langchain.docstore.document import Document

from ai_eval.resources.rag_prototype import (  # type: ignore[import-untyped]
    QADataItem,
    RAGConfig,
    RAGPrototype,
)


class _FakeLLM:
    """Minimal LLM stub that records prompts and returns deterministic answers."""

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        answer = f"[FAKE ANSWER] {prompt.splitlines()[-1][:120]}"
        return SimpleNamespace(content=answer)


class _FakeVectorStore:
    """In-memory store that keeps track of indexed nodes for assertions."""

    def __init__(self) -> None:
        self.nodes: List["_FakeTextNode"] = []

    def store_nodes(self, nodes: List["_FakeTextNode"]) -> None:
        self.nodes.extend(nodes)


class _FakeTextNode:
    """Replacement for LlamaIndex TextNode used in tests."""

    def __init__(self, text: str, metadata: Dict) -> None:
        self.text = text
        self.metadata = metadata
        self.score = 0.95

    def get_content(self) -> str:
        return self.text


class _FakeImageNode(_FakeTextNode):
    """Placeholder ImageNode so isinstance checks continue to work."""


class _FakeNodeParser:
    """Simple node parser that creates one node per document."""

    def __init__(self, chunk_size: int, chunk_overlap: int, paragraph_separator: str) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator

    def get_nodes_from_documents(self, docs: List) -> List[_FakeTextNode]:
        nodes: List[_FakeTextNode] = []
        for idx, doc in enumerate(docs):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata.setdefault("source", metadata.get("source", "doc"))
            metadata.setdefault("page", metadata.get("page", idx + 1))
            nodes.append(_FakeTextNode(text=getattr(
                doc, "text", doc.page_content), metadata=metadata))
        return nodes


class _FakeVectorStoreIndex:
    """Stores nodes inside the fake vector store for later retrieval."""

    def __init__(self, nodes: List[_FakeTextNode], storage_context: SimpleNamespace, show_progress: bool = False) -> None:
        self._nodes = list(nodes)
        storage_context.vector_store.store_nodes(nodes)


class _FakeRetrievedNode:
    """Wrapper returned by the fake retriever to mimic llama-index nodes."""

    def __init__(self, node: _FakeTextNode) -> None:
        self._node = node
        self.metadata = node.metadata
        self.score = node.score

    def get_content(self) -> str:
        return self._node.get_content()


class _FakeVectorIndexRetriever:
    """Retriever that simply returns the indexed nodes."""

    def __init__(self, index: _FakeVectorStoreIndex, similarity_top_k: int) -> None:
        self.index = index
        self._similarity_top_k = similarity_top_k

    def retrieve(self, question: str) -> List[_FakeRetrievedNode]:
        _ = question
        top_nodes = self.index._nodes[: self._similarity_top_k]
        return [_FakeRetrievedNode(node) for node in top_nodes]


class _FakeLangChainLLM:
    """Thin wrapper to mirror LangChainLLM interface used in RAGPrototype."""

    def __init__(self, llm: _FakeLLM) -> None:
        self.llm = llm

    def invoke(self, prompt: str) -> SimpleNamespace:
        return self.llm.invoke(prompt)


class _FakeRetrieverQueryEngine:
    """Query engine that concatenates retrieved content and calls the fake LLM."""

    def __init__(self, retriever: _FakeVectorIndexRetriever, llm: _FakeLLM) -> None:
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def from_args(cls, retriever: _FakeVectorIndexRetriever, llm, response_mode: str):
        _ = response_mode
        # LangChainLLM wraps ChatAnthropic in production. Here we unwrap it if needed.
        underlying_llm = getattr(llm, "llm", llm)
        return cls(retriever=retriever, llm=underlying_llm)

    def query(self, question: str) -> str:
        nodes = self.retriever.retrieve(question)
        context = "\n".join(node.get_content() for node in nodes)
        response = self.llm.invoke(f"Question: {question}\nContext: {context}")
        return response.content


class _FakeQdrantClient:
    """Placeholder Qdrant client to satisfy configuration requirements."""

    def __init__(self, url: str, api_key: str) -> None:
        self.url = url
        self.api_key = api_key

    def get_collections(self) -> SimpleNamespace:
        return SimpleNamespace(collections=[])

    def delete_collection(self, collection_name: str) -> None:
        _ = collection_name

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        _ = collection_name
        return SimpleNamespace(
            points_count=0,
            config=SimpleNamespace(params=SimpleNamespace(
                vectors=SimpleNamespace(size=2048))),
            status=SimpleNamespace(value="green"),
        )


class _FakeLlamaDocument:
    """Replacement for LlamaIndexDocument used inside RAGPrototype."""

    def __init__(self, text: str, metadata: Dict) -> None:
        self.text = text
        self.page_content = text  # For compatibility with LangChain Document conversion
        self.metadata = metadata


@pytest.fixture
def patched_rag(monkeypatch) -> Tuple[RAGPrototype, _FakeVectorStore]:
    """Patch heavy external dependencies so we can exercise the real pipeline logic."""
    from ai_eval.resources import rag_prototype as rag_module

    # Ensure RAGPrototype uses the fake helpers defined above.
    monkeypatch.setattr(rag_module, "TextNode", _FakeTextNode)
    monkeypatch.setattr(rag_module, "ImageNode", _FakeImageNode)
    monkeypatch.setattr(rag_module, "LlamaIndexDocument", _FakeLlamaDocument)
    monkeypatch.setattr(rag_module, "VectorStoreIndex", _FakeVectorStoreIndex)
    monkeypatch.setattr(
        rag_module, "VectorIndexRetriever", _FakeVectorIndexRetriever
    )
    monkeypatch.setattr(
        rag_module, "RetrieverQueryEngine", _FakeRetrieverQueryEngine
    )
    monkeypatch.setattr(rag_module, "LangChainLLM", _FakeLangChainLLM)
    monkeypatch.setattr(
        rag_module, "Settings", SimpleNamespace(embed_model=None)
    )

    monkeypatch.setattr(rag_module.qdrant_client,
                        "QdrantClient", _FakeQdrantClient)
    monkeypatch.setattr(rag_module, "JinaEmbedding",
                        lambda *args, **kwargs: SimpleNamespace())

    fake_vector_store = _FakeVectorStore()

    def _fake_create_llm(self) -> _FakeLLM:
        llm = _FakeLLM()
        self._test_llm = llm  # type: ignore[attr-defined]
        return llm

    def _fake_setup_vector_store(self):
        self.vector_store = fake_vector_store
        self.storage_context = SimpleNamespace(vector_store=fake_vector_store)
        self._is_deployed = False

    def _fake_node_parser(self):
        return _FakeNodeParser(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            paragraph_separator=self.config.paragraph_separator,
        )

    monkeypatch.setattr(RAGPrototype, "_create_llm", _fake_create_llm)
    monkeypatch.setattr(
        RAGPrototype,
        "_create_embedding_model",
        lambda self: SimpleNamespace(embed_dim=self.config.embedding_dim),
    )
    monkeypatch.setattr(
        RAGPrototype,
        "_create_qdrant_client",
        lambda self: _FakeQdrantClient(
            url="http://fake-qdrant", api_key="fake-key"),
    )
    monkeypatch.setattr(RAGPrototype, "_setup_vector_store",
                        _fake_setup_vector_store)
    monkeypatch.setattr(RAGPrototype, "_create_node_parser",
                        lambda self: _fake_node_parser(self))

    config = RAGConfig(
        qdrant_url="http://fake-qdrant",
        qdrant_api_key="fake-key",
        jina_api_key="fake-key",
        anthropic_api_key="fake-key",
        collection_name="pytest_collection",
        force_recreate=True,
        top_k=2,
    )

    rag = RAGPrototype(config=config)
    return rag, fake_vector_store


def _build_documents() -> List[Document]:
    """Helper to create deterministic LangChain documents for indexing."""
    return [
        Document(
            page_content=(
                "AEC specifications require reinforced concrete walls to meet fire safety "
                "ratings defined in Section 5.2 of the handbook."
            ),
            metadata={"source": "aec_specs.pdf", "page": 1},
        ),
        Document(
            page_content=(
                "Building managers should reference the inspection checklist in Appendix C "
                "to validate site safety compliance."
            ),
            metadata={"source": "site_safety.pdf", "page": 12},
        ),
    ]


def test_rag_pipeline_generates_dataset_file(patched_rag, tmp_path: Path):
    """RAGPrototype should index documents, answer questions, and emit QA JSON."""
    rag, fake_vector_store = patched_rag
    documents = _build_documents()

    metadata = rag.deploy_embeddings(documents)

    # Vector store should contain the indexed nodes so retrieval can work.
    assert rag.is_deployed is True
    assert len(fake_vector_store.nodes) == len(documents)
    assert metadata is not None
    assert metadata.num_nodes == len(fake_vector_store.nodes)

    result = rag.retrieve_structured(
        "How do we verify site safety compliance?", top_k=2)
    assert result.num_retrieved == 2
    assert result.has_scores
    assert any(
        "inspection checklist" in doc.page_content for doc in result.documents)

    qa_item = rag.generate_qa_data(
        "How do we verify site safety compliance?", index=7)
    assert isinstance(qa_item, QADataItem)
    assert qa_item.context != ""
    assert "inspection checklist" in qa_item.context

    output_path = tmp_path / "generated_qa_data_tum.json"
    rag.save_qa_dataset([qa_item], output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert len(saved) == 1
    saved_item = saved[0]

    expected_fields = {
        "index",
        "question",
        "answer",
        "location_dependency_evaluator_target_answer",
        "context",
        "groundedness_score",
        "groundedness_eval",
        "question_relevancy_score",
        "question_relevancy_eval",
        "faithfulness_score",
        "faithfulness_eval",
    }
    assert expected_fields.issubset(saved_item.keys())

    # The saved answer should reflect the fake LLM output and match retrieval.
    assert "[FAKE ANSWER]" in saved_item["answer"]
    assert "inspection checklist" in saved_item["context"]
