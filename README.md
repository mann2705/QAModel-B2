# QAModel-B2

# GROUP 1:

The uploaded files on Git are: 

1. Original DataSet:  "COVID-QA_Dataset.csv"
2. Preprocessed and new dataset:   "formatted_output.csv"
3. Python file:   "Practicum_B2.ipynb"


Tasks Performed:

1. The original dataset included two unnecessary columns which were dropped and only two columns question and answer were used.
2. After which a format for the LLM output was decided and a new dataset in that format was created in the name "formatted_output.csv" 
3. The output of the final LLM response should be of the similar format shown in the csv file "formatted_output.csv" in the form:
        Question: ............
        Answer: ..............

4. Tags were generated for the new dataset "formatted_output.csv" and divided into four clusters and according to which each pair of question and answer were classified and given tag number based on the cluster numbers. Use these tags to build the model architecture and perform the further steps on the updated dataset "formatted_output.csv"


Expected results:

The final LLM model should be given a question as a prompt related to covid and the output should be an answer for the same. If the question is not related to Covid then the answer should be that "please ask a covid related question" or "I am not able to answer questions outside of the covid domain" or any similar type of response.


Format for the generated output: 

### Question: What is covid-19?
### I am a covid Q&A Agent... 

### Answer: "Answer should be printed related to the question"



## Group 2

### Files on Git:
- Group2.py

### Tasks Performed:

1. Performed tokenization and created a vocabulary.
2. Developed a basic structure of a Transformer model.
3. Used a Naive Bayes Classifier for text classification.

   #Group 3
Tasks Performed:
Selection of Model Type: Decided on an encoder-decoder (BERT-like) model, as it is suitable for understanding tasks like question answering and classification.

Hyperparameter Selection:

Embedding Dimension: Chose the embedding dimension based on the bert-base-uncased model, which has a default embedding size of 768.
Number of Layers: Used a 12-layer model (BERT-base configuration).
Attention Heads: Set to 12 attention heads, which enables capturing multiple relationships between tokens.
Batch Size and Sequence Length:
Batch size set to 16 to accommodate Colab memory constraints.
Sequence length set to 512, allowing the model to handle the maximum input length for BERT-based tasks.
Model Initialization:

Weight Initialization: Utilized pre-trained weights from the bert-base-uncased model, which provides a solid starting point for fine-tuning on the COVID-QA task.
Token Embedding Initialization: Leveraged pre-trained embeddings from BERT (bert-base-uncased), which are optimized for understanding language context, especially suited for question-answering tasks.
Data Preparation:

Loaded and preprocessed the formatted_output.csv file by extracting questions and answers into separate columns.
Created a PyTorch dataset to tokenize the question-answer pairs using the BERT tokenizer.
Split the dataset into training and testing sets and set up DataLoaders for both.
Model Training Setup:

Configured the BERT model for sequence classification using the question-answer pairs as inputs.
Set up an optimizer (AdamW) and defined a basic training loop for fine-tuning the model.
Expected Results:

The model should respond to COVID-related questions with relevant answers based on fine-tuning.
If a non-COVID-related question is asked, the model will respond with a predefined message indicating that it only answers COVID-related questions.

