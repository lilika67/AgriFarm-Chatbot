# AgriFarm Chatbot README

## Overview

This project implements a chatbot for agricultural queries using the AgroQA dataset. The chatbot leverages natural language processing (NLP) techniques and a transformer-based model to provide answers to farming-related questions. The Jupyter Notebook (`AgriFarm_Chatbot.ipynb`) contains the code for data preprocessing, exploratory data analysis, and model training.

## Technologies Used

The following technologies and libraries are used in this project:
- **Python**: The primary programming language (version 3.11 or later).
- **Jupyter Notebook**: For interactive development and execution of the codebase.
- **Pandas**: For data loading, manipulation, and preprocessing of the AgroQA dataset.
- **NLTK (Natural Language Toolkit)**: For text preprocessing tasks such as tokenization, lemmatization, and stopword removal.
- **Transformers (Hugging Face)**: For leveraging pre-trained transformer models (e.g., T5 or BERT) for natural language understanding and generation.
- **Datasets (Hugging Face)**: For efficient dataset handling and preparation for model training.
- **Sacrebleu**: For evaluating model performance using BLEU scores.
- **Gradio**: For creating an interactive web-based interface for the chatbot.
- **Matplotlib**: For data visualization and exploratory data analysis.
- **Regular Expressions (re)**: For text normalization (e.g., removing punctuation).
- **GPU Acceleration**: Utilized for faster model training (requires CUDA-compatible hardware).

## Dataset Description

We use the dataset from Kaggle(https://www.kaggle.com/datasets/jonathanomara/agronomic-question-and-answer-dataset).The AgroQA dataset (`AgroQA Dataset.csv`) contains 3,044 entries with the following columns:
- **Crop**: The type of crop related to the question (e.g., wheat, rice).
- **Question**: The agricultural query posed by the user.
- **Answer**: The corresponding answer to the question.
- **input_text**: A derived column combining the question and crop context (e.g., "question: [Question] context: [Crop]").
- **target_text**: The cleaned and normalized answer text used for training.

### Dataset Preprocessing

- **Loading**: The dataset is loaded using pandas from `AgroQA Dataset.csv`.
- **Cleaning**: Missing values are removed (1 missing answer was dropped), and duplicates are eliminated (8 duplicates removed, resulting in 3,036 entries).
- **Normalization**: Text in `input_text` and `target_text` columns is converted to lowercase, and punctuation is removed using regular expressions.
- **Tokenization**: NLTK is used for tokenization, with stop words and other NLP preprocessing steps applied to prepare the data for model training.

## Performance Metrics

- **BLEU Score**: Measures the similarity between generated and reference answers, using the `sacrebleu` library.
- **ROUGE Score**: Evaluates overlap in n-grams, word sequences, and word pairs between generated and reference texts.
- **Perplexity**: Assesses the model's language modeling performance (lower is better).
- **User Satisfaction**: Qualitative feedback from user interactions via the Gradio interface.

To evaluate the chatbot, you can extend the notebook to include these metrics after training a model (e.g., using a transformer like T5 or BERT).

## Chat Interaction Screenshots

This releted screenshots of interactions with the chatbot via the Gradio interface. 

![page with user instruction on how to use AgriFarm chatbot](<img width="950" alt="Screenshot 2025-06-22 at 10 56 16" src="https://github.com/user-attachments/assets/92dcf6b4-14d6-4156-8dce-7c493823de40" />
)
![Chat Interaction Examples1](<img width="1221" alt="Screenshot 2025-06-22 at 19 03 51" src="https://github.com/user-attachments/assets/f73827df-4268-4226-89d8-f56c0572faf6" />
)
![Chat Interaction Examples2](<img width="1221" alt="Screenshot 2025-06-22 at 19 04 05" src="https://github.com/user-attachments/assets/d29d75c0-e596-408f-b8d7-7ff537e3c6e4" />
)
![Chat Interaction Examples3](<img width="1221" alt="Screenshot 2025-06-22 at 19 04 12" src="https://github.com/user-attachments/assets/4306e704-8183-47ee-a58c-71add1a243ac" />
)
![Chat Interaction Examples4](<img width="1221" alt="Screenshot 2025-06-22 at 19 04 18" src="https://github.com/user-attachments/assets/a7aaf0b8-eab6-4a45-8649-09ad660bec4d" />
)
![Chat Interaction when user asks irrevant questions](<img width="1221" alt="Screenshot 2025-06-22 at 19 04 27" src="https://github.com/user-attachments/assets/23fbea93-c41d-4f04-8776-94019393db12" />
)


## Requirements
The following Python packages are required to run the chatbot:
- `transformers`
- `datasets`
- `sacrebleu`
- `gradio`
- `nltk`
- `pandas`
- `matplotlib`

These are installed via the following commands in the notebook:
```bash
!pip install transformers datasets sacrebleu gradio --quiet
!pip install nltk
```

Additionally, NLTK data resources (`punkt`, `wordnet`, `stopwords`, `punkt_tab`) are downloaded for text preprocessing.

## Steps to Run the Chatbot

Follow these steps to set up and run the chatbot:

1. **Set Up Environment**:
   - Ensure you have Python 3.11 or later installed.
   - Install the required packages:
     ```bash
     pip install transformers datasets sacrebleu gradio nltk pandas matplotlib
     ```
   - Download NLTK data:
     ```python
     import nltk
     nltk.download('punkt')
     nltk.download('wordnet')
     nltk.download('stopwords')
     nltk.download('punkt_tab')
     ```

2. **Prepare the Dataset**:
   - Place the `AgroQA Dataset.csv` file in the same directory as the `AgriFarm_Chatbot.ipynb` notebook.
   - The dataset should have columns: `Crop`, `Question`, and `Answer`.

3. **Run the Jupyter Notebook**:
   - Open the `AgriFarm_Chatbot.ipynb` notebook in Jupyter Notebook or JupyterLab.
   - Execute the cells in order to:
     - Install dependencies.
     - Load and preprocess the dataset.
     - Perform exploratory data analysis (EDA).
     - Visualize data 

4. **Train the Model** 
   

5. **Launch the Chatbot Interface**:
   
   - Run the cell containing the Gradio interface code to launch the chatbot in a web browser.

6. **Interact with the Chatbot**:
   - Input a question (e.g., "What is the best fertilizer for soil?") a in the Gradio interface.
   - The chatbot will respond based on the trained model and the AgroQA dataset.

## Notes

- **Hugging Face Deployment**: The trained model has been deployed on Hugging Face. You can access it via the Hugging Face Model Hub  `https://huggingface.co/kaytesi/t5-finetunedModel').
- **GPU Acceleration**: The notebook is configured to use a GPU (`accelerator: GPU`) or even CPU but we recommend using GPU for faster training.
- **Dataset Availability**: The `AgroQA Dataset.csv` file must be accessible. If not available, replace it with a similar dataset or generate synthetic data.
- **Extending the Notebook**: To complete the chatbot, implement model training and inference sections. Use the `transformers` library for fine-tuning and the Gradio interface for user interaction.
- **Visualization**: The notebook includes a plot (not fully shown) for data visualization. Ensure `matplotlib` is installed to view these plots.

## Troubleshooting

- **Missing Dataset**: If `AgroQA Dataset.csv` is missing, the notebook will fail to load the data. Ensure the file is in the correct directory.
- **NLTK Errors**: If NLTK resources fail to download, retry the `nltk.download()` commands or check your internet connection.
- **Gradio Interface**: If the Gradio interface does not launch, ensure the Gradio package is installed and check for port conflicts.
- **GPU Issues**: If GPU is unavailable, modify the notebook to run on CPU by removing the `accelerator: GPU` setting or using a compatible environment (e.g., Google Colab with GPU runtime).

For further details or issues, refer to the documentation of the used libraries (`transformers`, `gradio`, `nltk`) or contact the project maintainer.

## Author

**Liliane Kayitesi**
