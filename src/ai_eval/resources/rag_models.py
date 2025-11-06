from typing import Union, List, Literal, Optional
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_google_vertexai import ChatVertexAI
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from ai_eval.config.config import rag_model_list
from ai_eval.config import global_config as glob
from ai_eval.services.logger import LoggerFactory

my_logger = LoggerFactory(handler_type="Stream", verbose=True).create_module_logger()


class RAGModels:

    def __init__(
        self,
        experiment_type: str,
        group: str,
        docs: List[Document],
    ) -> None:
        """
        Initialize the RAG model with specified configuration.

        Args:
            experiment_type (str): Type of experiment ('generation' or 'retriever').
            group (str): Model group identifier.
            docs (List[Document]): List of documents for vectorstore initialization.

        Raises:
            ValueError: If model provider is not supported.
        """
        print(experiment_type, group)
        models = rag_model_list[experiment_type]
        key = list(models[group].keys())[0]

        if experiment_type == 'generation' and group == 'embedding_model':
            experiment_type = 'retriever'

        params = models[group]["hyperparameters"]
        if params is None:
            params = {}

        if experiment_type == 'retriever':
            from typing import Any

            match key:
                case "google":
                    model: Any = VertexAIEmbeddings(
                        project=glob.GCP_PROJECT,
                        model_name=models[group][key],
                        **params,
                    )
                case "ollama":
                    embedding_model_name = models[group][key]
                    model: Any = OllamaEmbeddings(model=embedding_model_name, **params)
                case _:
                    raise ValueError(f"Model provider {key} not supported.")

            my_logger.info("Building FAISS local vectorstore...")
            self.vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=model,
            )
            my_logger.info("Vectorstore built.")
        elif experiment_type == 'generation':
            # Select the chat/generation and judge models
            match key:
                case "google":
                    self.model = ChatVertexAI(
                        project=glob.GCP_PROJECT,
                        model_name=models[group][key],
                        **params,
                    )
                case "ollama":
                    self.model = ChatOllama(model_name=models[group][key], **params)
                case _:
                    raise ValueError(
                        f"Model provider {glob.MODEL_PROVIDER} not supported."
                    )

            my_logger.info("Models loaded successfully")

    def callRetrieverModel(self, query: str) -> str:
        """
        Retrieve relevant documents based on the input query using similarity search.

        Args:
            query (str): The search query to find relevant documents.

        Returns:
            str: A concatenated string of the most relevant document contents.

        Note:
            Uses FAISS similarity search to find the top 5 most relevant documents
            from the vectorstore.
        """
        relevant_docs = self.vectorstore.similarity_search(query=query, k=5)
        matched_docs = " ".join(doc.page_content for doc in relevant_docs)
        # matched_docs = [doc.page_content for doc in relevant_docs]
        return matched_docs

    def callLLMModel(self, query: str, context: str) -> str:
        """
        Generate an answer using the LLM model based on the given query and context.

        Args:
            query (str): The question to be answered.
            context (str): The context information to help generate the answer.

        Returns:
            BaseMessage: The model's response containing the generated answer.

        Note:
            Uses a RAG (Retrieval-Augmented Generation) approach where the model:
            1. Takes the provided context and question
            2. Generates a comprehensive but concise answer
            3. References source document numbers when relevant
            4. Returns no answer if the information cannot be found in the context
        """
        rag_prompt = """
            Using the information contained in the context,
            give a comprehensive answer to the question.
            Respond only to the question asked, response should be concise and relevant to the question.
            Provide the number of the source document when relevant.
            If the answer cannot be deduced from the context, do not give an answer.
            Context:
            {context}
            ---
            Now here is the question you need to answer.
            Question: {question}
            Answer:
            """

        rag_generator_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=rag_prompt,
        )
        final_prompt = rag_generator_prompt.format(context=context, question=query)
        answer = self.model.invoke(final_prompt)
        return answer
