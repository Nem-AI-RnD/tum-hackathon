# Application for RAG test data generation

This repo contains Python code to build and deploy a Streamlit UI on GCP Cloud run where users can generate a synthetically annotated validation set for the evaluation of a RAG application. In the UI a user can upload a new PDF file (max. 200 MB) to Google Cloud storage (currently only PDFs with valid text layers are supported, i.e. no OCR support!) and can then use their document to generate (Question/Answer/Context) tuples along with two quality scores to measure the quality of the generation task: Question Groundedness, Question Relevancy and Answer Faithfulness (where 1 means very bad and 5 very good quality). This approach is called 'LLM-as-a-judge' in the literature and is briefly outlined below.

The main data generation (or better 'annotation') task happens in another Cloud run service which exposes a REST API (see *synth_app.py*). This service is being called by the UI once a user requests a new data set generation. All API requests are written to a Firestore backend with User ID, request date-time, status and document name. 

Watch the <a href="https://onenemetschek-my.sharepoint.com/:v:/r/personal/fmazza_nemetschek_com/Documents/Team%20Shared%20Folder/Projects/Nemy%20-%20AI%20assistant/nemy-test-data-generator-rag-app.mov?csf=1&web=1&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=1ZckKj" target="_blank">video</a> for a live demo of the application.

For local development either selected models via the Vertex AI SDK can be used or selected Ollama models, see [model list](src/ai_eval/config/model_config.yaml). 


## LLM-as-a-judge:

**Reference:** <a href="https://arxiv.org/abs/2306.05685" target="_blank">*Zheng et al. (2023): Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*</a>. Based on this <a href="https://lmsys.org/blog/2023-03-30-vicuna/" target="_blank">blog</a>.

The main idea is to rather than let a human expert annotate / label a validation data set for information retrieval (i.e. rate the retrieved context + answer given a query) it is more cost efficient to utilize current LLMs to do that job. 

- Select a context and ask an LLM to generate (Question / Answer) - pair 
  - Context --> (Query / Answer)
    
- Ask (another) LLM ('Judge') to evaluate each (Question / Answer / Context) – triple
  - Question Groundedness
  - Question relevancy
  - (Answer) Faithfulness

Next create 'predictions' (for context / answer) using your RAG system. Note: we now have (synthetic) ground truth for 'Answer / Context - given Question'.   

- RAG: Query --> (Context / Answer)
  - Evaluate Retrieval (i.e. Context) + Generation (i.e. Answer) task separately   
  - Use e.g. DeepEval for this; e.g. Answer Faithfulness

<br>

**The approach can be depicted as follows:**

<br> 

<img src="logos/llm_judge.drawio.png" alt="architecture" width="700" height="400">


## Package structure

```
├── CHANGELOG.md
├── data
├── Docker
│   ├── Dockerfile_rag
│   ├── Dockerfile_streamlit
│   └── Dockerfile_synthesizer
├── docker-compose.yml
├── logos
├── Makefile
├── pyproject.toml
├── rag_app.py
├── README.md
├── src
│   ├── ai_eval
│   │   ├── config
│   │   ├── resources
│   │   ├── services
│   │   └── utils
│   └── notebooks
├── synth_app.py
├── tests
├── upload_ui.py
```

## Package installation and application develoment

Create virtual environment: 
```bash
uv venv --python 3.12
uv sync
# source .venv/bin/activate
```

Install dev-dependencies:
```bash
uv pip install ".[dev]"
```

## Build API + UI images
```bash
make up
```

## Local testing - start all services
```bash
make all-services
```


```bash
curl --location 'http://127.0.0.1:8000/create_data' \
--header 'Content-Type: application/json' \
--data '{
  "blob_input_path": "raw_documents/20240731_Nemetschek SE_Mitarbeiterhandbuch_Employee Handbook.pdf",
  "blob_output_path": "mytest.csv",
  "n_generations": 5
}'
```

## ToDos - next steps

- Try speeding up the API call by converting synchronous to asynchronous calls where possible

- Add chat history (requests + answers) to Firestore, maybe add new collection

- Add user login + authentication

- Add st.download_button to download generated data set

- Add unit tests for methods

- Add third tab for RAG calling on the generated test set

- Create KPIs based on DeepEval or RAGAS (?)

- Push created (annotated) test data to LangFuse automatically as part of e2e pipeline



