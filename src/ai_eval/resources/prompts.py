# QA_generation_prompt = """
# Your task is to write a factoid question and an answer based on the given context.
# The factoid question should be answerable with a specific, concise piece of factual information from the context.
# Formulate the factoid question in the same style as questions users might ask in a search engine.
# Do NOT include phrases like "according to the passage" or "context" in your question.
# Always check the language of the provided context and write the question and your answer in the same language.
# Only use english if the context is in english.

# Provide your output in the following format:
# Factoid question: <factoid question>
# Answer: <answer to the factoid question; Maximum 300 characters long>

# Here is the context: {context}\n
# Output:
# """

QA_generation_prompt = """
You are an advanced language model and multilingual expert in BIM software and the AEC domain, tasked with generating one question-and-answer pair for a user-uploaded document chunk, simulating a chat bot GUI interaction about BIM software functionality or tasks (e.g., modelling, clash detection, project management).
 
## Instructions
 
1. **Language**: Detect the language of the document chunk and generate the Q&A in that language only, without translation.
 
2. **Analysis**: Identify a key BIM software feature or instruction in the chunk, focusing on common AEC tasks. Use the conversation history summary to ensure the question is relevant, non-redundant, and aligns with prior context.
 
3. **Q&A Generation**:
   - Create one specific factoid question (max 300 characters) as if you were a BIM software user, answerable from the chunk.
   - Ensure the question reflects a practical BIM task (e.g., “How do I run a clash test?”).
   - Provide a concise answer (max 300 characters), derived directly from the chunk, avoiding references to specific document locations (e.g., “page 5”).
   - Avoid vague or context-dependent questions requiring additional user information not available here (e.g. which tabs a hypothetical user may have opened, cursor position).
 
4. **Q/A Examples for positive case (no additional/external info needed) + negative case (vice versa)**:
   - **Positive**:  
     **Chunk Context**: When creating new tender / LV information, such as structures, positions, notes, etc., the application distinguishes the possible commands depending on the standard (ÖNORM or GAEB). 
     **Question**: According to which standards does the application distinguish commands when creating tender / LV information?  
     **Answer**: The application differentiates commands depending on the standard (ÖNORM or GAEB).
   - **Negative**:  
     **Chunk Context**: Deductions and withdrawals: For special cases (e.g. construction cleaning), you can enter additional withdrawals/withdrawals from net or gross here. 
       These amounts are listed in the compilation of the report and in the dialog according to the "net amount of services provided" or the gross amount. 
       The "Apply from" selection box accesses the options listed in Manage | Master data projects defined sets of payments and withdrawals in order to transfer them to the order. 
       The following entries are available here: Description: Designation of the deduction/purchase that is output in the print output. 
       Subtotal: When printing out the compilation, a subtotal is formed, which can later be used to calculate withdrawals. 
       Basis of calculation: Reference to a subtotal (see also identifier) Percent / Amount: Percentage value or amount and type of withdrawal/withdrawal (cover relinquishment, security retention, contract performance bond, contractual withdrawal/withdrawal, free withdrawal/withdrawal) Identifier: Only possible if the subtotal is active (ticked box). Serves as an identifier that can be referred to in further lines via the Calculation Basis field.
       This can be a subtotal immediately before it, which is then called ZWN1+N2 as an identifier at this point. 
       In the calculation basis, one then refers to ZWN1+N2 one line later and possibly calculates a -10.00% discount from it. For invoices only.  
     **Question**: What is the identifier used for?  
     **Answer**: Sorry, this question cannot be answered properly without additional contextual user information.
 
5. **Scope**: Generate one Q&A pair per chunk, addressing a distinct BIM software aspect, avoiding overlap with prior interactions.
 
6. **Tone**: Use a professional, clear, approachable tone for BIM users. Explain technical terms briefly if needed, avoiding undefined jargon.
 
7. **Constraints**:
   - Base Q&A solely on the chunk {context} without assuming external information.
   - Avoid speculative answers or content not in the chunk.
   - Ensure questions are specific and answerable without user location or additional data.
 
8. **Output Format**:
   - Present the Q&A pair in the following format:  
     Factoid question: <question>
     Answer: <answer, max 300 characters>
   - Ensure the question is phrased as if asked in a chat bot GUI.
"""

QA_generation_prompt_with_memory = """
You are an advanced language model and multilingual expert in BIM software and the AEC domain, tasked with generating one question-and-answer pair for a user-uploaded document chunk, simulating a chat bot GUI interaction about BIM software functionality or tasks (e.g., modelling, clash detection, project management).
 
## Instructions
 
1. **Language**: Detect the language of the document chunk and generate the Q&A in that language only, without translation.
 
2. **Analysis**: Identify a key BIM software feature or instruction in the chunk, focusing on common AEC tasks. Use the conversation history summary to ensure the question is relevant, non-redundant, and aligns with prior context.
 
3. **Q&A Generation**:
   - Create one specific factoid question (max 300 characters) as if you were a BIM software user, answerable from the chunk.
   - Ensure the question reflects a practical BIM task (e.g., “How do I run a clash test?”).
   - Provide a concise answer (max 300 characters), derived directly from the chunk, avoiding references to specific document locations (e.g., “page 5”).
   - Avoid vague or context-dependent questions requiring additional user information not available here (e.g. which tabs a hypothetical user may have opened, cursor position).
 
4. **Q/A Examples for positive case (no additional/external info needed) + negative case (vice versa)**:
   - **Positive**:  
     **Chunk Context**: When creating new tender / LV information, such as structures, positions, notes, etc., the application distinguishes the possible commands depending on the standard (ÖNORM or GAEB). 
     **Question**: According to which standards does the application distinguish commands when creating tender / LV information?  
     **Answer**: The application differentiates commands depending on the standard (ÖNORM or GAEB).
   - **Negative**:  
     **Chunk Context**: Deductions and withdrawals: For special cases (e.g. construction cleaning), you can enter additional withdrawals/withdrawals from net or gross here. 
       These amounts are listed in the compilation of the report and in the dialog according to the "net amount of services provided" or the gross amount. 
       The "Apply from" selection box accesses the options listed in Manage | Master data projects defined sets of payments and withdrawals in order to transfer them to the order. 
       The following entries are available here: Description: Designation of the deduction/purchase that is output in the print output. 
       Subtotal: When printing out the compilation, a subtotal is formed, which can later be used to calculate withdrawals. 
       Basis of calculation: Reference to a subtotal (see also identifier) Percent / Amount: Percentage value or amount and type of withdrawal/withdrawal (cover relinquishment, security retention, contract performance bond, contractual withdrawal/withdrawal, free withdrawal/withdrawal) Identifier: Only possible if the subtotal is active (ticked box). Serves as an identifier that can be referred to in further lines via the Calculation Basis field.
       This can be a subtotal immediately before it, which is then called ZWN1+N2 as an identifier at this point. 
       In the calculation basis, one then refers to ZWN1+N2 one line later and possibly calculates a -10.00% discount from it. For invoices only.  
     **Question**: What is the identifier used for?  
     **Answer**: Sorry, this question cannot be answered properly without additional contextual user information.
 
5. **Scope**: Generate one Q&A pair per chunk, addressing a distinct BIM software aspect, avoiding overlap with prior interactions.
 
6. **Tone**: Use a professional, clear, approachable tone for BIM users. Explain technical terms briefly if needed, avoiding undefined jargon.
 
7. **Constraints**:
   - Base Q&A solely on the chunk {context} and history {history}, without assuming external information.
   - Avoid speculative answers or content not in the chunk.
   - Ensure questions are specific and answerable without user location or additional data.
 
8. **Output Format**:
   - Present the Q&A pair in the following format:  
     Factoid question: <question>
     Answer: <answer, max 300 characters>
   - Ensure the question is phrased as if asked in a chat bot GUI.
"""

qa_prompt_template = """
You are an expert question-answer pair generator.
Given the document chunk below, generate one high-quality factoid question-answer pair in the same language as the input chunk.
Detect the language of the chunk and ensure both the question and answer are in that language.
The factoid question should be concise and directly derived from the chunk’s content (e.g., who, what, where, when).
The answer must be concise, accurate, and no longer than 300 characters.

Input:
A document chunk {text}

Provide your output in the following format:
Factoid question: <factoid question>
Answer: <answer to the factoid question; Maximum 300 characters long>
Output:
"""


answer_location_dependency_prompt = """
You are an advanced language model and expert evaluator in BIM software and the Architecture, Engineering, and Construction (AEC) domain, tasked with evaluating a single question-and-answer pair from a BIM software user manual chunk. Evaluate whether the **question** is clear, specific, and fully answerable based on the provided context. Assign a binary score: 0 if the question is well-formed, clear, and can be answered completely using the context without needing additional document references (e.g., page, section, chapter) or clarification; 1 if the question is vague, incomplete, or requires additional context or document references to be answerable.

## Evaluation Criteria
- **Score 0**: The question is clear, specific, and directly answerable using the provided context, with no need for additional document location references or clarification.
- **Score 1**: The question is vague, ambiguous, or lacks sufficient detail to be answered fully based on the context, requiring document location references (e.g., page, section) or rephrasing for clarity.
- Check if the question’s phrasing, scope, or assumptions make it difficult to provide a complete answer without further context or clarification.

## Examples
- **Positive (Score 0): No improvement needed with location info):  
  **Context**: When creating new tender / LV information, such as structures, positions, notes, etc., the application distinguishes the possible commands depending on the standard (ÖNORM or GAEB).  
  **Q&A Pair**:  
    Factoid question: According to which standards does the application distinguish commands when creating tender / LV information?  
    Answer: The application differentiates commands depending on the standard (ÖNORM or GAEB).  
  **Reasoning**: The question is clear, specific, and directly answerable using the context without needing further clarification.  
  **Score**: 0  
- **Negative Example** (Score 1: Could be improved with location info):  
  **Context**: Deductions and withdrawals: For special cases (e.g. construction cleaning), you can enter additional withdrawals/withdrawals from net or gross here. These amounts are listed in the compilation of the report and in the dialog according to the "net amount of services provided" or the gross amount. The "Apply from" selection box accesses the options listed in Manage | Master data projects defined sets of payments and withdrawals in order to transfer them to the order. The following entries are available here: Description: Designation of the deduction/purchase that is output in the print output. Subtotal: When printing out the compilation, a subtotal is formed, which can later be used to calculate withdrawals. Basis of calculation: Reference to a subtotal (see also identifier) Percent / Amount: Percentage value or amount and type of withdrawal/withdrawal (cover relinquishment, security retention, contract performance bond, contractual withdrawal/withdrawal, free withdrawal/withdrawal) Identifier: Only possible if the subtotal is active (ticked box). Serves as an identifier that can be referred to in further lines via the Calculation Basis field. This can be a subtotal immediately before it, which is then called ZWN1+N2 as an identifier at this point. In the calculation basis, one then refers to ZWN1+N2 one line later and possibly calculates a -10.00% discount from it. For invoices only.  
  **Q&A Pair**:  
    Factoid question: What is the identifier used for?  
    Answer: Sorry, this question cannot be answered properly without additional contextual user information.  
  **Reasoning**: The question is vague, requiring clarification or document reference to identifier 
  **Score**: 1  

## Instructions
- Provide concise, objective reasoning for the score (max 200 characters), referencing the criteria.
- Always assign a binary score (0 or 1).
- If the score is 1, **always generate a refined question** that is clearer, more specific, or better aligned with the context to achieve a score of 0, in the language used in the user's input or context and **provide an answer to the refined question** based on the context and in the same language as the context. If the score is 0, provide empty "Refined Question" and "Refined Answer" fields.
- Output the response in JSON format, including the question, answer, context, and evaluation.
- Further If the score is 1, **always generate as target answer** as "This question is hard to answer accurately as it seems to depend on some more context. Could you please provide me with more context, e.g what you are currently doing in the software?". If the score is 0, provide the inputted answer as the target answer.

## Output Format:
   - Question: {question}\n
   - Answer: {answer} \n
   - Context: {context}\n
   Please structure your response in the following JSON format.
   Please always directly output the json object, without any additional text or comments.

   ```json
  {{
        "Evaluation": "<reasoning for the rating>",
        "Score": "<integer rating either 0 or 1>",
	      "Refined Question": "<refined question if score is 1; empty string if score is 0>",
        "Refined Answer": "<refined answer if score is 1; empty string if score is 0>",
        "Target Answer": "<target answer if score is 1; empty string if score is 0>
  }}
  ```
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


question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' or scoring how well one can answer the given question based on the given context.
Rate your answer on an integer scale of 1 to 5, based on the following score rubric:

Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

### Score Rubric:
Score 1: The question is not answerable at all given the context. The context does not provide any relevant information to answer the question.
Score 2: The question is barely answerable given the context. The context provides minimal relevant information, but the answer would be mostly incorrect, inaccurate, or not factual.
Score 3: The question is somewhat answerable given the context. The context provides some relevant information, but the answer would be partially correct, accurate, and factual.
Score 4: The question is mostly answerable given the context. The context provides sufficient relevant information, and the answer would be mostly correct, accurate, and factual.
Score 5: The question is clearly and unambiguously answerable given the context. The context provides all the necessary relevant information, and the answer would be completely correct, accurate, and factual.

In any case, you must provide a score!

Question: {question}\n
Context: {context}\n

Please structure your response in the following JSON format.
Please always directly output the json object, without any additional text or comments.

```json
{{
        "Evaluation": "<reasoning for the rating>",
        "Score": "<integer rating between 1 and 5>"
}}
```
\n\nAnswer: <json_output>
"""

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to employees at the Nemetschek Group SE.
Rate the usefulness of the question on an integer scale of 1 to 5, based on the following score rubric:

### Score Rubric:
Score 1: The question is not useful at all. It does not provide any relevant information or value to the employees.
Score 2: The question is barely useful. It provides minimal relevant information, but its value to the employees is limited.
Score 3: The question is somewhat useful. It provides some relevant information, but its value to the employees is moderate.
Score 4: The question is mostly useful. It provides sufficient relevant information and is valuable to the employees.
Score 5: The question is extremely useful. It provides all the necessary relevant information and is highly valuable to the employees.

In any case, you must provide a score!

Question: {question}\n

Please structure your response in the following JSON format.
Please always directly output the json object, without any additional text or comments.

```json
{{
        "Evaluation": "<reasoning for the rating>",
        "Score": "<integer rating between 1 and 5>"
}}
```
\n\nAnswer: <json_output>
"""


answer_faithfulness_critique_prompt = """
You will be given a context, question, and an answer generated by the system.
Your task is to assess the quality of the answer based on its faithfulness to the context.

### Score Rubric:
Score 1: The answer is completely unfaithful (contains many inaccuracies or information not in the context)
Score 2: The answer is mostly unfaithful (has significant deviations from the context)
Score 3: The answer is somewhat faithful (has minor deviations but generally follows the context)
Score 4: The answer is mostly faithful (accurately reflects the context with minimal deviations)
Score 5: The answer is completely faithful (all information accurately reflects the context)

Context: {context}
Question: {question}
Answer: {answer}

Please structure your response in the following JSON format.
Please always directly output the json object, without any additional text or comments.

{{
    "Evaluation": "<detailed explanation of how faithfully the answer reflects the context>",
    "Score": "<integer rating between 1 and 5>"
}}
\n\nAnswer: <json_output>
"""


answer_relevancy_critique_prompt = """
You will be given a question and an answer generated by the system.
Your task is to assess how closely the answer addresses the query. Consider whether the response directly answers the question, stays on topic, and provides relevant details that pertain to the query.
Rate the relevancy on a scale of 1 to 5, where:
- 1 means the answer is not relevant at all (it does not address the question or is completely off-topic),
- 5 means the answer is fully relevant (it directly addresses the question with appropriate details).

Provide your answer as follows:

Answer:::
Evaluation: (explain your reasoning for the rating, citing specific examples of how the answer addresses or fails to address the question)
Total rating: (your rating, as a number between 1 and 5)

Now here are the question and answer.

Question: {question}\n
Answer: {answer}\n
Answer:::"""
