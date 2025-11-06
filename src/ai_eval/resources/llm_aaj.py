import random

# import asyncio
from typing import Dict, List, Optional, Tuple, Union

from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from tqdm.auto import tqdm

from ai_eval.config import global_config as glob
from ai_eval.config.config import model_list
from ai_eval.resources.data_schemas import OutputStructure
from ai_eval.resources.prompts import rag_prompt
from ai_eval.services.logger import LoggerFactory
from ai_eval.utils.utils import timer

my_logger = LoggerFactory().create_module_logger()

sampled_contexts = None


@timer
def build_local_vectorstore(
    chunked_docs: List[Document],
) -> VectorStore:
    """
    Loads embeddings for a list of already chunked documents and returns a vector store.

    Args:
        chunked_docs (List[Document]): A list of documents to be embedded

    Returns:
        VectorStore: A vector store containing the embedded documents.
    """
    # Filter out documents with empty or missing content
    valid_docs = [doc for doc in chunked_docs if doc.page_content.strip()]
    if not valid_docs:
        raise ValueError("No valid documents to process. All documents are empty.")

    my_logger.info(
        f"Processing {len(valid_docs)} valid out of {len(chunked_docs)} documents for vectorstore."
    )

    # Embedding models available
    selected_models = model_list["embedding_model"][glob.MODEL_PROVIDER]
    embedding_model_name = selected_models[list(selected_models.keys())[0]]

    embedding_model: Union[VertexAIEmbeddings, OllamaEmbeddings]
    match glob.MODEL_PROVIDER:
        case "google":
            embedding_model = VertexAIEmbeddings(
                project=glob.GCP_PROJECT, model_name=embedding_model_name
            )
        case "ollama":
            embedding_model = OllamaEmbeddings(model=embedding_model_name)
        case _:
            raise ValueError(f"Model provider {glob.MODEL_PROVIDER} not supported.")

    my_logger.info("Building FAISS local vectorstore...")
    vectorstore = FAISS.from_documents(
        documents=valid_docs,
        embedding=embedding_model,
    )
    my_logger.info("Vectorstore built.")
    return vectorstore


def answer_with_rag(
    question: str,
    llm: LLM,
    vectorstore: VectorStore,
) -> Tuple[str, List[Document]]:
    """
    Generate an answer to a question using Retrieval-Augmented Generation (RAG).
    This function uses a combination of document retrieval and language model generation
    to produce a concise answer to the given question. It optionally reranks the retrieved
    documents to improve the relevance of the final answer.
    Args:
        question (str): The question to be answered.
        llm (LLM): The language model to generate the final answer.
        vectorstore (VectorStore): The vector store used for document retrieval.
    Returns:
        Tuple[str, List[Document]]: A tuple containing the generated answer and the list of relevant documents.
    """

    rag_retrieval_prompt = PromptTemplate(
        input_variables=["context"],
        template="You are an AI document retrieval assistant. \
            Using the following context generate a concise answer: {context}\n\n \
            Output also the source of the information. \
            \n\nAnswer:",
    )

    rag_generator_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=rag_prompt,
    )
    # 1.) Build the retriever part
    retriever = vectorstore.as_retriever()
    combine_docs_chain = create_stuff_documents_chain(llm, rag_retrieval_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    retrieval_result = retrieval_chain.invoke({"input": question})

    answer = retrieval_result["answer"]
    relevant_docs = retrieval_result["context"]

    # Ensure relevant_docs is iterable
    if relevant_docs is None:
        relevant_docs = []

    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f" Document No. {str(i)}: " + doc for i, doc in enumerate(relevant_docs)]
    )
    # 2.) Build the generator part:
    final_prompt = rag_generator_prompt.format(context=context, question=question)
    answer = llm.invoke(final_prompt)

    return answer, relevant_docs


# def judge_llm(prompt: PromptTemplate, llm: LLM, **inputs: str) -> OutputStructure:
#     """
#     Executes a judgment process using a provided prompt template, language model (LLM),
#     and additional input parameters. The function utilizes a JSON output parser to
#     structure the output into a specified Pydantic object.

#     Args:
#         prompt (PromptTemplate): The template for the prompt to be used in the judgment process.
#         llm (LLM): The language model to process the prompt and generate a response.
#         **inputs (str): Additional keyword arguments representing input variables for the prompt.

#     Returns:
#         OutputStructure: A structured output parsed into the specified Pydantic object.
#     """
#     parser = JsonOutputParser(pydantic_object=OutputStructure)
#     judge_chain = (prompt | llm | parser).with_config({"callbacks": [opik_tracer]})
#     return judge_chain.invoke(inputs)


def judge_llm(prompt: PromptTemplate, llm: LLM, **inputs: str) -> OutputStructure:
    """
    Executes a judgment process using a provided prompt template, language model (LLM),
    and additional input parameters. The function utilizes a JSON output parser to
    structure the output into a specified Pydantic object.

    Args:
        prompt (PromptTemplate): The template for the prompt to be used in the judgment process.
        llm (LLM): The language model to process the prompt and generate a response.
        **inputs (str): Additional keyword arguments representing input variables for the prompt.

    Returns:
        OutputStructure: A structured output parsed into the specified Pydantic object.
    """
    parser = JsonOutputParser(pydantic_object=OutputStructure)
    judge_chain = prompt | llm | parser
    return judge_chain.invoke(inputs)


# def chat_llm(prompt: PromptTemplate, llm: LLM, **inputs: str) -> str:
#     """
#     Call a language model with a given prompt template and inputs.

#     This function processes a prompt through a language model and returns the response as a string.

#     Args:
#         prompt (PromptTemplate): The template to format with input variables
#         llm (LLM): The language model to process the prompt
#         **inputs: Variable keyword arguments to populate the prompt template

#     Returns:
#         str: The processed response from the language model

#     Example:
#         >>> prompt = PromptTemplate("Answer this question: {question}")
#         >>> llm = ChatOpenAI()
#         >>> response = chat_llm(prompt, llm, question="What is 2+2?")
#     """

#     question_router = (prompt | llm | StrOutputParser()).with_config(
#         {"callbacks": [opik_tracer]}
#     )
#     return question_router.invoke(inputs)


def chat_llm(prompt: PromptTemplate, llm: LLM, **inputs: str) -> str:
    """
    Call a language model with a given prompt template and inputs.

    This function processes a prompt through a language model and returns the response as a string.

    Args:
        prompt (PromptTemplate): The template to format with input variables
        llm (LLM): The language model to process the prompt
        **inputs: Variable keyword arguments to populate the prompt template

    Returns:
        str: The processed response from the language model

    Example:
        >>> prompt = PromptTemplate("Answer this question: {question}")
        >>> llm = ChatOpenAI()
        >>> response = chat_llm(prompt, llm, question="What is 2+2?")
    """

    question_router = prompt | llm | StrOutputParser()
    return question_router.invoke(inputs)


async def chat_llm_async(prompt: PromptTemplate, llm: LLM, **inputs: str) -> str:
    """
    Call a language model asynchronously with a given prompt template and inputs.

    This function processes a prompt through a language model and returns the response as a string.

    Args:
        prompt (PromptTemplate): The template to format with input variables
        llm (LLM): The language model to process the prompt
        **inputs: Variable keyword arguments to populate the prompt template

    Returns:
        str: The processed response from the language model
    """
    question_router = prompt | llm | StrOutputParser()
    res = await question_router.ainvoke(inputs)
    return res


@timer
def _get_sampled_contexts(
    docs_processed: List[Document], n_generations: int, with_replacement: bool
) -> List[Document]:
    """
    Samples a specified number of documents from a list, either with or without replacement.

    Args:
        docs_processed (List[Document]): The list of documents to sample from.
        n_generations (int): The number of documents to sample.
        with_replacement (bool): Whether to sample with replacement (True) or without replacement (False).

    Returns:
        List[Document]: A list of sampled documents.
    """

    my_logger.info(f"Total documents {len(docs_processed)} ")
    if with_replacement:
        sampled_contexts = random.choices(docs_processed, k=n_generations)
        my_logger.info(f"Sampling {len(sampled_contexts)} contexts with replacement")
    else:
        sampled_contexts = random.sample(
            docs_processed, min(len(docs_processed), n_generations)
        )
        my_logger.info(f"Sampling {len(sampled_contexts)} contexts without replacement")
    return sampled_contexts


def generate_qa_couples(
    docs_processed: List[Document],
    prompt: PromptTemplate,
    llm: LLM,
    n_generations: int = 10,
    with_replacement: bool = False,
) -> List[dict]:
    """
    Generate QA couples from the given documents.

    Args:
        docs_processed (list): List of processed documents.
        prompt (PromptTemplate): The prompt template to generate QA couples.
        n_generations (int): Number of QA couples to generate.

    Returns:
        list: List of generated QA couples.
    """
    my_logger.info(f"Generating {n_generations} QA couples...")

    outputs = []
    sampled_contexts = _get_sampled_contexts(
        docs_processed, n_generations, with_replacement
    )

    for idx, sampled_context in enumerate(tqdm(sampled_contexts)):

        # Generate QA couple
        # --------------------
        output_QA_couple = chat_llm(
            prompt=prompt, llm=llm, context=sampled_context.page_content
        )

        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split(
                "Answer: "
            )[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            assert len(answer) < 500, f"Answer is too long in QA couple {idx}!"
            outputs.append(
                {
                    "context": sampled_context.page_content,
                    "question": question.replace("\n", ""),
                    "answer": answer.replace("\n", ""),
                    "source_doc": sampled_context.metadata["source"],
                    "index": idx,
                }
            )
        except Exception as e:
            my_logger.error(f"Error processing context at index {idx}: {e}")
            continue
    return outputs


def eval_qa_couples(
    outputs: List[dict],
    prompt1: PromptTemplate,
    prompt2: PromptTemplate,
    prompt3: PromptTemplate,
    prompt4: PromptTemplate,
    llm: LLM,
) -> List[dict]:
    """
    Rate each QA couple based on groundedness and question relevancy.

    Args:
        outputs: List of dictionaries containing QA couples to rate
        prompt1: Prompt template for groundedness evaluation
        prompt2: Prompt template for question relevancy evaluation
        prompt3: Prompt template for answer faithfulness
        llm: Language model to use for rating, i.e. the judge

    Returns:
        List of dictionaries with added rating scores and evaluations
    """
    print(
        "Rating each QA couple according to groundedness, question relevancy and answer faithfulness..."
    )
    # print("outputs",outputs)
    for output in tqdm(outputs, total=len(outputs), desc="Evaluating QA couples"):
        evaluations = {
            "groundedness": judge_llm(
                prompt=prompt1,
                llm=llm,
                question=output["question"],
                context=output["context"],
            ),
            "question_relevancy": judge_llm(
                prompt=prompt2, llm=llm, question=output["question"]
            ),
            "faithfulness": judge_llm(
                prompt=prompt3,
                llm=llm,
                question=output["question"],
                answer=output["answer"],
                context=output["context"],
            ),
            "location_dependency_evaluator": judge_llm(
                prompt=prompt4,
                llm=llm,
                question=output["question"],
                answer=output["answer"],
                context=output["context"],
            ),
        }
        # Print some output
        print("\n================================")
        print(output["question"])
        # print(output["context"][:150])
        print(output["answer"])
        print("================================")

        # print(evaluations["groundedness"])
        # print(evaluations["question_relevancy"])
        # print(evaluations["faithfulness"])
        # print("================================\n")

        try:
            for criterion, evaluation in evaluations.items():
                # print(f"\n--- {criterion} evaluation ---")
                # print(evaluation)
                if criterion == 'location_dependency_evaluator':
                    output.update(
                        {
                            f"{criterion}_score": int(evaluation["Score"]),
                            f"{criterion}_eval": evaluation["Evaluation"],
                            f"{criterion}_question": evaluation["Refined Question"],
                            f"{criterion}_answer": evaluation["Refined Answer"],
                            f"{criterion}_target_answer": evaluation["Target Answer"],
                        }
                    )
                else:
                    output.update(
                        {
                            f"{criterion}_score": int(evaluation["Score"]),
                            f"{criterion}_eval": evaluation["Evaluation"],
                        }
                    )
        except Exception as e:
            print(f"Error processing evaluation for criterion {criterion}: {e}")
            continue
    return outputs


async def answer_with_rag_async(
    question: str,
    llm: LLM,
    vectorstore: VectorStore,
) -> Tuple[str, List[Document]]:
    """
    Generate an answer to a question using Retrieval-Augmented Generation (RAG) asynchronously.
    This function uses a combination of document retrieval and language model generation
    to produce a concise answer to the given question.

    Args:
        question (str): The question to be answered.
        llm (LLM): The language model to generate the final answer.
        vectorstore (VectorStore): The vector store used for document retrieval.

    Returns:
        Tuple[str, List[Document]]: A tuple containing the generated answer and the list of relevant documents.
    """

    rag_retrieval_prompt = PromptTemplate(
        input_variables=["context"],
        template="You are an AI document retrieval assistant. \
            Using the following context generate a concise answer: {context}\n\n \
            Output also the source of the information. \
            \n\nAnswer:",
    )

    rag_generator_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Using the following context, answer the question: {context}\n\nQuestion: {question}\n\nAnswer:",
    )

    # 1.) Build the retriever part
    retriever = vectorstore.as_retriever()
    combine_docs_chain = create_stuff_documents_chain(llm, rag_retrieval_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Use asyncio to retrieve documents asynchronously
    retrieval_result = await retrieval_chain.ainvoke({"input": question})

    answer = retrieval_result["answer"]
    relevant_docs = retrieval_result["context"]

    # Ensure relevant_docs is iterable
    if relevant_docs is None:
        relevant_docs = []

    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f" Document No. {str(i)}: " + doc for i, doc in enumerate(relevant_docs)]
    )

    # 2.) Build the generator part:
    final_prompt = rag_generator_prompt.format(context=context, question=question)

    # Use asyncio to invoke the language model asynchronously
    answer = await llm.ainvoke(final_prompt)

    return answer, relevant_docs


def answer_with_rag_tfidf(
    question: str, llm: LLM, documents: Optional[List[Document]], k: int = 3
) -> tuple[str, list[Document]]:
    """
    Answers a question using Retrieval-Augmented Generation (RAG) with TF-IDF document retrieval.

    This function retrieves relevant documents from the provided list using TF-IDF similarity to the question,
    constructs a context from the retrieved documents, and generates an answer using the provided language model (LLM).
    The answer is constrained to use only facts from the retrieved context.

    Args:
        question (str): The question to be answered.
        llm (LLM): The language model to generate the answer. Must implement an `invoke` method.
        documents (Optional[List[Document]]): A list of documents to retrieve context from. Each document should have a `page_content` attribute.
        k (int): The number of top relevant documents to retrieve.

    Returns:
        Tuple[str, List[Document]]: A tuple containing the generated answer string and the list of relevant documents used as context.
    """

    if documents is None:
        raise ValueError("You must provide a list of documents for TFIDF retrieval.")

    # If documents are Document objects, extract text
    if hasattr(documents[0], "page_content"):
        texts: List[str] = [doc.page_content for doc in documents]
    else:
        texts: List[str] = documents  # type: ignore

    retriever = TFIDFRetriever.from_texts(texts)
    relevant_docs = retriever.invoke(question, k=k)
    k = min(k, len(relevant_docs))
    relevant_docs = relevant_docs[:k]

    context = " ".join([doc.page_content for doc in relevant_docs])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer with facts from the context only."),
            ("human", "{input}\nContext: {context}"),
        ]
    )

    str_parser = StrOutputParser()
    prompted = prompt.invoke({"input": question, "context": context})
    answer = str_parser.invoke(llm.invoke(prompted))
    return answer, relevant_docs


# @timer
def generate_qa_memory(
    docs_processed: List[Document],
    prompt: PromptTemplate,
    llm: LLM,
    n_generations: int = 10,
    sampled_contexts: Optional[List[Document]] = None,
    with_replacement: bool = False,
) -> List[Dict[str, str]]:
    """
    Generates a list of QA (question-answer) pairs using a language model with conversation memory.

    Args:
        docs_processed (List[Document]): List of processed documents to sample contexts from.
        prompt (PromptTemplate): Prompt template to guide the LLM in generating QA pairs.
        llm (LLM): Language model instance used for generating responses.
        n_generations (int, optional): Number of QA pairs to generate. Defaults to 10.
        sampled_contexts (Optional[List[Document]], optional): Pre-sampled contexts to use for generation. If None, contexts are sampled from docs_processed. Defaults to None.
        with_replacement (bool, optional): Whether to sample contexts with replacement. Defaults to False.

    Returns:
        List[dict]: A list of dictionaries, each containing:
            - "context": The text context used for QA generation.
            - "question": The generated factoid question.
            - "answer": The generated answer.
            - "source_doc": Source document metadata.
            - "index": Index of the QA pair.
    """

    my_logger.info(f"Generating {n_generations} QA couples with memory...")

    # Initialize conversation memory
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=2000,
        # memory_key="chat_history",
        return_messages=True,
    )

    qa_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)

    outputs = []
    sampled_contexts = _get_sampled_contexts(
        docs_processed, n_generations, with_replacement
    )

    for idx, sampled_context in enumerate(tqdm(sampled_contexts)):
        # my_logger.debug(f"Sampled context length {len(sampled_context.page_content)}")

        try:
            output = qa_chain.predict(context=sampled_context.page_content)

            if not output.strip():
                raise ValueError("Empty LLM output")

            question = output.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output.split("Answer: ")[-1]
            assert len(answer) < 500, f"Answer is too long in QA couple {idx}!"
            outputs.append(
                {
                    "context": sampled_context.page_content,
                    "question": question.replace("\n", ""),
                    "answer": answer.replace("\n", ""),
                    "source_doc": sampled_context.metadata["source"],
                    "index": idx,
                }
            )
        except Exception as e:
            my_logger.error(f"Error processing context at index {idx}: {e}")
            continue
    return outputs
