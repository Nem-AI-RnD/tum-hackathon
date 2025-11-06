import os
from langchain_ollama.chat_models import ChatOllama
from langchain_google_vertexai import ChatVertexAI
from langchain_community.document_loaders import PyPDFLoader

from ai_eval.services.file_gcp import stream_gcs_pdf, CSVService

# from ai_eval.services.file import CSVService
from ai_eval.services.file_gcp import PDFService, JSONService
from ai_eval.services.clients import GCPClient
from ai_eval.utils.utils import list_objects
from ai_eval.resources.preprocessor import Preprocessor
from ai_eval.services.database import FirestoreService
from ai_eval.resources.predictor import Predictor
from ai_eval.resources.synthesizer import TestDatasetCreator, Synthesizer
from ai_eval.config.config import model_list
from ai_eval.config import global_config as glob
from ai_eval.resources.get_models import InitModels

filename = "Allplan_2020_Manual.pdf"

if __name__ == "__main__":
    # pre = Preprocessor()
    # docs = pre.fetch_documents(
    #     blob_path=f"{glob.DATA_PKG_DIR}/{filename}", source="local"
    # )
    # chunks = pre.chunk_documents(documents=docs)
    models = InitModels()

    synth = Synthesizer(
        qa_generator=models.qa_generator,
        eval_model=models.eval_model,
        blob_path=f"{glob.DATA_PKG_DIR}/{filename}",
        source="local",
    )

    out = synth.create_qa_data(
        n_generations=5,
    )


pred = Predictor()

# chunked_docs = pred.load_test_data()

deep_data = pred.predict()
print(deep_data)

# pred.save_deepeval_dataset()

pred.upload_to_opik(dataset_name="My Test Dataset")

# filename = "20240731_Nemetschek SE_Mitarbeiterhandbuch_Employee Handbook.pdf"

# loader = PyPDFLoader(f"{glob.DATA_PKG_DIR}/{filename}")

# raw_data = loader.load()

models = InitModels()

synth = Synthesizer(
    qa_generator=models.qa_generator,
    eval_model=models.eval_model,
    blob_path="raw_documents/Allplan_2020_Manual.pdf",
)

out = synth.create_qa_data(
    n_generations=20,
)

# synth.save_data()

synth.upload_to_opik(
    dataset_name="My Test Dataset",
)

# from ai_eval.services.database import FirestoreService

# Initialize the Document class with a collection name
docs = FirestoreService(collection_name="test_collection")

# docs.delete_collection()

# docs.collection_name

# # Create a new document
# new_doc_id = docs.create({"title": "New Document", "content": "This is the content."})

# # Read the document
# doc_data = docs.read(new_doc_id)
# doc_data
# # Update the document
# docs.update(new_doc_id, {"content": "Updated content"})

# # Delete the document
# docs.delete(new_doc_id)

# # List documents
# document_list = docs.list_documents(limit=5)
# document_list

# # Query documents
# query_results = docs.query_documents("title", "==", "New Document")

# ---------------------------------------------
list_objects("sandbox-eval")

# Example usage
# bucket_name = "sandbox-eval"
blob_path = (
    "raw_documents/20240731_Nemetschek SE_Mitarbeiterhandbuch_Employee Handbook.pdf"
)

# # CReate a pandas dataframe with 3 columns and 2 rows
# import pandas as pd
# from ai_eval.services.file_gcp import CSVService

# df = pd.DataFrame(
#     {
#         "A": [1, 2],
#         "B": [3, 4],
#         "C": [5, 6],
#     }
# )
# df

c = CSVService(root_path="raw_documents", path="generated_test_data.csv")
# c.doWrite(X=df)
data = c.doRead()
data.columns

contexts = data["context"].tolist()
generated_questions = data["question"].tolist()

contexts

from langchain.docstore.document import Document

# Convert the list of strings to a list of Document objects
docs_processed = [
    Document(page_content=context, metadata={"source": "generated_test_data.csv"})
    for context in data["context"].tolist()
]
docs_processed

from ai_eval.resources.llm_aaj import build_local_vectorstore, answer_with_rag

vectorstore = build_local_vectorstore(docs_processed)


# txt_service = TXTservice(
#     root_path="raw_documents",
#     path="test2.txt",
# )
# my_text = txt_service.doRead()

# # Models available
# google_chat = model_list["chat_model"].get(glob.MODEL_PROVIDER, "google")
# ollama_chat = model_list["chat_model"].get("ollama")

# # Select the chat/generation and judge models
# # qa_generator = ChatVertexAI(
# #     model_name=google_chat[list(google_chat.keys())[0]], temperature=0
# # )
# qa_generator = ChatOllama(model=ollama_chat[list(ollama_chat.keys())[0]], temperature=0)
# print(f"QA Generator: {qa_generator}")

# # rag_generator = ChatVertexAI(
# #     model_name=google_chat[list(google_chat.keys())[1]], temperature=0
# # )
# rag_generator = ChatOllama(
#     model=ollama_chat[list(ollama_chat.keys())[0]], temperature=0
# )

# print(f"RAG Generator: {rag_generator}")
# # eval_model = ChatVertexAI(
# #     model_name=google_chat[list(google_chat.keys())[1]], temperature=0
# # )
# eval_model = ChatOllama(model=ollama_chat[list(ollama_chat.keys())[0]], temperature=0.1)
# print(f"Evaluation Model: {eval_model}")

# deepl = DeepEvalTestDatasetCreator(
#     blob_path=blob_path,
#     qa_generator_llm=qa_generator,
#     eval_llm=eval_model,
#     rag_generator_llm=rag_generator,
# )

# documents = deepl.load_chunked_documents(blob_path=blob_path)

# import os
# from io import BytesIO
# from ai_eval.services.file_gcp import PDFService
# from ai_eval.config import global_config as glob

# pdf = PDFService(
#     root_path="raw_documents",
#     path="Allplan_2020_Manual_2.pdf",
# )

# # Create BytesIO object from local PDF
# local_pdf_path = os.path.join(
#     glob.DATA_PKG_DIR,
#     "Allplan_2020_Manual_TEST.pdf",
# )
# with open(local_pdf_path, 'rb') as file:
#     pdf_bytes = BytesIO(file.read())
# pdf_bytes

# pdf.doWrite(X=pdf_bytes)

# pdf.doRead(
#     local_pdf_path=os.path.join(
#         glob.DATA_PKG_DIR,
#         "Allplan_2020_Manual_TEST.pdf",
#     )
# )

# pd.doWrite(
#     local_pdf_path=os.path.join(
#         glob.DATA_PKG_DIR,
#         "20240731_Nemetschek SE_Mitarbeiterhandbuch_Employee Handbook.pdf",
#     )
# )

models = InitModels()

# Main Synthesizer
synth = Synthesizer(
    qa_generator=models.qa_generator,
    eval_model=models.eval_model,
    rag_generator=models.rag_generator,
    blob_path=blob_path,
)

dataset = synth.create_qa_data(
    n_generations=20,
)

generated_questions = synth.generated_questions

# generated_questions[
#     [
#         "question",
#         "answer",
#         "groundedness_score",
#         "question_relevancy_score",
#     ]
# ]

print(generated_questions["groundedness_score"].value_counts())

synth.save_data("mytest.csv")

# c = CSVService(path="this.csv")
# c.doWrite(X=dd)

# c = CSVService_GCP(root_path="raw_documents", path="this.csv")
# c.doWrite(X=dd)


# txt_service = TXTservice(
#     root_path="raw_documents",
#     path="test2.txt",
# )
# my_text = txt_service.doRead()
# print(my_text)
# txt_service.doWrite(X=["Hello World"])


# prep = Preprocessor()
# documents = prep.fetch_documents()
# documents

# len(documents)

# # Initialize GCPClient
# gcp_client = GCPClient(bucket_name)

# # Fetch blob content
# blob_content = gcp_client.download_blob(blob_path)

# Stream and process the PDF
streamed_text = stream_gcs_pdf(blob_path)


# # Print extracted text
# for page_text in streamed_text:
#     print(page_text)

# content = "".join([doc for doc in streamed_text])
# content
