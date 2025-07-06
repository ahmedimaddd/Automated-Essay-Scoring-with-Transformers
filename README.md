# Automated Essay Scoring using Transformers

This repository contains the code and findings for a comprehensive research project on Automated Essay Scoring (AES). The project evaluates and compares the performance of different transformer-based models (DeBERTa-v3 and DistilBERT) on the ASAP-AES dataset, using a robust K-Fold cross-validation methodology.

## Project Overview

The goal of this project was to build and rigorously evaluate a deep learning pipeline for automatically grading student essays. This involved several key stages:
1.  **Data Preprocessing:** Thoroughly cleaning and normalizing the essay text to prepare it for the models.
2.  **Data Augmentation:** Using `nlpaug` to synthetically increase the size and diversity of the training data.
3.  **Model Training:** Fine-tuning large language models using the Hugging Face `Trainer` API.
4.  **Robust Evaluation:** Implementing a 5-fold cross-validation loop to ensure the model's performance was stable and not dependent on a single train-test split.

## Key Achievements & Findings

* **High Performance:** Achieved a strong **Quadratic Weighted Kappa (QWK) score of 0.65** with an optimized DistilBERT model, demonstrating a high level of agreement with human graders.
* **Comparative Model Analysis:** Conducted a detailed analysis comparing the large `DeBERTa-v3-large` model with the more efficient `DistilBERT`.
* **Critical Insight:** Discovered that while powerful, the DeBERTa-v3 model consistently converged to a single prediction value, resulting in a QWK score of 0.0. The lighter DistilBERT model proved more effective, successfully capturing the ordinal nature of the scores.
* **Robust Methodology:** The use of K-Fold cross-validation provided a much more reliable measure of the models' true performance compared to a simple train-validation split.

## Technologies & Frameworks

* **Core Frameworks:** Hugging Face (`transformers`, `datasets`, `accelerate`), PyTorch
* **Models:** `microsoft/deberta-v3-large`, `distilbert-base-uncased`
* **Data Augmentation:** `nlpaug`
* **Text Processing:** `contractions`, `nltk`
* **Data Handling & ML:** Pandas, NumPy, Scikit-learn

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ahmedimaddd/Automated-Essay-Scoring-with-Transformers]
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download NLTK Data:**
    The `nlpaug` library requires certain NLTK resources. Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```
4.  **Run the notebook:**
    Open and run the Jupyter Notebook (`dibetra_3_Kfolds_methoddolgy) (1).ipynb`) to see the data preparation, training loop, and evaluation.

For a comprehensive analysis of the literature, methodology, and results, please see the full project report included in this repository.
