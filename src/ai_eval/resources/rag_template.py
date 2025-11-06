from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from ai_eval.utils.utils import validate_documents


class RAG(ABC):
    """Abstract base class for Retrieval-Augmented Generation (RAG) pipelines.

    Attributes:
        llm: Language model instance.
        documents: List of Document objects.
        k: Number of documents to retrieve.
        vectorstore: Optional vectorstore for retrieval (FAISS, etc.).
    """

    def __init__(
        self,
        llm: LLM,
        documents: List[Document] | None = None,
        k: int = 3,
        vectorstore: VectorStore | None = None,
    ) -> None:
        """Initialize RAG with a language model, documents, k value, and optional vectorstore.

        Args:
            llm: Language model instance.
            documents: List of Document objects (optional).
            k: Number of documents to retrieve (default: 3).
            vectorstore: Optional vectorstore for retrieval.
        """
        self.llm = llm
        self.documents = documents if documents is not None else []
        self.k = k
        self.vectorstore = vectorstore

    def _retrieve(
        self,
        question: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Document]:
        """Internal method to retrieve relevant documents for the given question.

        Args:
            question: Input question string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects relevant to the question.
        """
        docs = self.retrieve(question, *args, **kwargs)
        assert isinstance(docs, list), f"relevant_docs must be list, got {type(docs)}"
        validate_documents(docs)
        return docs

    @abstractmethod
    def retrieve(
        self,
        question: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Document]:
        """Implementation for retrieving relevant documents.

        Args:
            question: Input question string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects.
        """
        pass

    def _generate(
        self,
        question: str,
        context: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Internal method to generate an answer using the question and context.

        Args:
            question: Input question string.
            context: Context string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated answer string.
        """
        assert isinstance(context, str), f"context must be str, got {type(context)}"
        response = self.generate(question, context, *args, **kwargs)
        assert isinstance(response, str), f"response must be str, got {type(response)}"
        return response

    @abstractmethod
    def generate(
        self,
        question: str,
        context: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Implementation for generating an answer from question and context.

        Args:
            question: Input question string.
            context: Context string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated answer string.
        """
        pass

    def answer(
        self,
        question: str,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, List[Document]]:
        """Answer a question by retrieving documents, building context, and generating a response.

        Args:
            question: Input question string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (answer string, list of relevant Document objects).
        """
        assert isinstance(question, str), f"question must be str, got {type(question)}"
        relevant_docs = self._retrieve(question, *args, **kwargs)
        context = self.build_context(relevant_docs)
        response = self._generate(question, context, *args, **kwargs)
        return response, relevant_docs

    def build_context(self, relevant_docs: List[Document]) -> str:
        """Concatenate page content from relevant documents into a single context string.

        Args:
            relevant_docs: List of Document objects.

        Returns:
            Concatenated context string. Returns an empty string if no documents are provided.
        """
        if not relevant_docs:
            return ""
        return " ".join(
            [getattr(doc, "page_content", str(doc)) for doc in relevant_docs]
        )


# Some example implementations of RAG using TFIDF and FAISS:
# ----------------------------------------------------------
class TFIDFRAG(RAG):
    def __init__(self, llm: LLM, documents: List[Document], k: int = 3):
        super().__init__(llm, documents, k)
        # Extract texts from Document objects
        self.texts = [doc.page_content for doc in documents]
        self.retriever = TFIDFRetriever.from_texts(self.texts)

    def retrieve(self, question: str, *args, **kwargs) -> List[Document]:
        """Retrieve relevant documents for the given question."""
        relevant_docs = self.retriever.invoke(question, k=self.k)
        k = min(self.k, len(relevant_docs))
        return relevant_docs[:k]

    def generate(self, question: str, context: str, *args, **kwargs) -> str:
        """Generate an answer using the question and context."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer with facts from the context only."),
                ("human", "{input}\nContext: {context}"),
            ]
        )
        str_parser = StrOutputParser()
        answer = (prompt | self.llm | str_parser).invoke(
            {"input": question, "context": context}
        )
        return answer


class FAISSRAG(RAG):
    """RAG implementation using FAISS vectorstore for retrieval."""

    def __init__(
        self,
        llm: LLM,
        documents: List[Document],
        k: int = 3,
        vectorstore: VectorStore = None,
    ) -> None:
        super().__init__(llm, documents, k, vectorstore)

    def retrieve(
        self,
        question: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Document]:
        retriever = self.vectorstore.as_retriever()
        rag_retrieval_prompt = PromptTemplate(
            input_variables=["context"],
            template="You are an AI document retrieval assistant. Using the following context generate a concise answer: {context}\n\nOutput also the source of the information.\n\nAnswer:",
        )
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, rag_retrieval_prompt
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        retrieval_result = retrieval_chain.invoke({"input": question})
        relevant_docs = retrieval_result["context"]
        if relevant_docs is None:
            return []
        return relevant_docs

    def generate(
        self,
        question: str,
        context: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        rag_generator_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Using the following context, answer the question: {context}\n\nQuestion: {question}\n\nAnswer:",
        )
        final_prompt = rag_generator_prompt.format(context=context, question=question)
        answer = self.llm.invoke(final_prompt)
        # Ensure the output is a string
        if hasattr(answer, "content"):
            return answer.content
        return str(answer)
