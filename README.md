AgriFarm Chatbot README
Overview
This project implements a chatbot for agricultural queries using the AgroQA dataset. The chatbot leverages natural language processing (NLP) techniques and a transformer-based model to provide answers to farming-related questions. The Jupyter Notebook (AgriFarm_Chatbot.ipynb) contains the code for data preprocessing, exploratory data analysis, and model training.
Dataset Description
The AgroQA dataset (AgroQA Dataset.csv) contains 3,044 entries with the following columns:

Crop: The type of crop related to the question (e.g., wheat, rice).
Question: The agricultural query posed by the user.
Answer: The corresponding answer to the question.
input_text: A derived column combining the question and crop context (e.g., "question: [Question] context: [Crop]").
target_text: The cleaned and normalized answer text used for training.

Dataset Preprocessing

Loading: The dataset is loaded using pandas from AgroQA Dataset.csv.
Cleaning: Missing values are removed (1 missing answer was dropped), and duplicates are eliminated (8 duplicates removed, resulting in 3,036 entries).
Normalization: Text in input_text and target_text columns is converted to lowercase, and punctuation is removed using regular expressions.
Tokenization: NLTK is used for tokenization, with stop words and other NLP preprocessing steps applied to prepare the data for model training.

Performance Metrics
The notebook does not explicitly include model training or evaluation code in the provided snippet, but typical performance metrics for a chatbot of this nature include:

BLEU Score: Measures the similarity between generated and reference answers, using the sacrebleu library.
ROUGE Score: Evaluates overlap in n-grams, word sequences, and word pairs between generated and reference texts.
Perplexity: Assesses the model's language modeling performance (lower is better).
User Satisfaction: Qualitative feedback from user interactions via the Gradio interface.

To evaluate the chatbot, you can extend the notebook to include these metrics after training a model (e.g., using a transformer like T5 or BERT).
Requirements
The following Python packages are required to run the chatbot:

transformers
datasets
sacrebleu
gradio
nltk
pandas
matplotlib (for visualizations)

These are installed via the following commands in the notebook:
!pip install transformers datasets sacrebleu gradio --quiet
!pip install nltk

Additionally, NLTK data resources (punkt, wordnet, stopwords, punkt_tab) are downloaded for text preprocessing.
Steps to Run the Chatbot
Follow these steps to set up and run the chatbot:

Set Up Environment:

Ensure you have Python 3.11 or later installed.
Install the required packages:pip install transformers datasets sacrebleu gradio nltk pandas matplotlib


Download NLTK data:import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')




Prepare the Dataset:

Place the AgroQA Dataset.csv file in the same directory as the AgriFarm_Chatbot.ipynb notebook.
The dataset should have columns: Crop, Question, and Answer.


Run the Jupyter Notebook:

Open the AgriFarm_Chatbot.ipynb notebook in Jupyter Notebook or JupyterLab.
Execute the cells in order to:
Install dependencies.
Load and preprocess the dataset.
Perform exploratory data analysis (EDA).
Visualize data (if applicable).




Train the Model (if implemented):

The provided notebook snippet focuses on preprocessing and EDA. To train a model, add code to fine-tune a transformer model (e.g., T5 or BERT) using the transformers library on the input_text and target_text columns.
Example training setup (not included in the snippet but recommended):from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
# Add training code here




Launch the Chatbot Interface:

The notebook uses Gradio to create an interactive interface. Ensure the Gradio interface code is implemented (not shown in the provided snippet).
Example Gradio setup:import gradio as gr
def chatbot(question, crop):
    input_text = f"question: {question} context: {crop}"
    # Add model inference code here
    return "Sample response"
gr.Interface(fn=chatbot, inputs=["text", "text"], outputs="text").launch()


Run the cell containing the Gradio interface code to launch the chatbot in a web browser.


Interact with the Chatbot:

Input a question (e.g., "What is the best fertilizer for wheat?") and the crop type (e.g., "Wheat") in the Gradio interface.
The chatbot will respond based on the trained model and the AgroQA dataset.



Notes

GPU Acceleration: The notebook is configured to use a GPU (accelerator: GPU). Ensure you have a compatible GPU and CUDA installed if running on a local machine.
Dataset Availability: The AgroQA Dataset.csv file must be accessible. If not available, replace it with a similar dataset or generate synthetic data.
Extending the Notebook: To complete the chatbot, implement model training and inference sections. Use the transformers library for fine-tuning and the Gradio interface for user interaction.
Visualization: The notebook includes a plot (not fully shown) for data visualization. Ensure matplotlib is installed to view these plots.

Troubleshooting

Missing Dataset: If AgroQA Dataset.csv is missing, the notebook will fail to load the data. Ensure the file is in the correct directory.
NLTK Errors: If NLTK resources fail to download, retry the nltk.download() commands or check your internet connection.
Gradio Interface: If the Gradio interface does not launch, ensure the Gradio package is installed and check for port conflicts.
GPU Issues: If GPU is unavailable, modify the notebook to run on CPU by removing the accelerator: GPU setting or using a compatible environment (e.g., Google Colab with GPU runtime).

For further details or issues, refer to the documentation of the used libraries (transformers, gradio, nltk) or contact the project maintainer.
