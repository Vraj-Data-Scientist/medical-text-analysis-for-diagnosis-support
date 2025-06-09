# Medical Text Analysis for Diagnosis Support

## Overview
This project implements a machine learning pipeline to classify medical transcriptions into medical specialties, aiding clinical decision support and EHR organization. It uses both statistical (Logistic Regression) and transformer-based (BioBERT) models on the `mtsamples.csv` dataset.

## Features
- **Data Preprocessing**: Handles missing values, combines transcription and keywords, and applies domain-specific text cleaning (e.g., preserving medical keywords, expanding abbreviations).
- **Text Augmentation**: Uses SynonymAug to balance minority classes.
- **Statistical Model**: Logistic Regression with TF-IDF, SMOTE, and feature selection.
- **Transformer Model**: Fine-tuned BioBERT for medical specialty classification.
- **Evaluation**: Accuracy, Macro/Micro F1 scores, and confusion matrices.

---

### Problem in the Domain

| **Aspect**                | **Description**                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| **Domain**                | Healthcare, specifically medical text analysis for clinical decision support.    |
| **Problem**               | Manual classification of medical transcriptions into specialties is time-consuming, error-prone, and requires expert knowledge. |
| **Challenges**            | - High variability in medical terminology and writing styles.<br>- Class imbalance in medical specialties.<br>- Need for high accuracy due to clinical implications.<br>- Handling noisy, unstructured text data. |
| **Impact**                | Misclassification can lead to delays in diagnosis, inappropriate treatments, or increased workload for healthcare professionals. |

---

### What This Project Does

| **Aspect**                | **Description**                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| **Objective**             | Automatically classify medical transcriptions into one of 13 medical specialties (after combining low-frequency classes). |
| **Purpose**               | To assist healthcare providers by automating the categorization of medical reports, improving efficiency, and supporting diagnosis workflows. |
| **Applications**          | - Clinical decision support systems.<br>- Electronic health record (EHR) organization.<br>- Medical research and data mining. |

---

### How This Project Works

| **Stage**                 | **Details**                                                                     |
|---------------------------|---------------------------------------------------------------------------------|
| **Data Preprocessing**    | - Loads `mtsamples.csv` dataset.<br>- Combines transcription and keywords, handles missing values.<br>- Combines low-frequency specialties into an "others" class.<br>- Applies text preprocessing (lowercasing, removing punctuation/numbers, expanding abbreviations, lemmatization, stopword removal while preserving medical terms). |
| **Text Augmentation**     | Uses SynonymAug (WordNet) to augment minority classes to address class imbalance. |
| **Data Preparation**      | - Splits data into train (75%) and test (25%) sets with stratification.<br>- Encodes labels using LabelEncoder. |
| **Statistical Model**     | - Uses TF-IDF vectorization (15,000 features, n-grams 1-4).<br>- Applies SMOTE for class imbalance.<br>- Selects top 10,000 features using chi-squared.<br>- Trains Logistic Regression with GridSearchCV (C, penalty, solver). |
| **Transformer Model**     | - Uses BioBERT (dmis-lab/biobert-base-cased-v1.1).<br>- Fine-tunes for 10 epochs with class-weighted loss.<br>- Uses a custom BERTDataset class and DataLoader.<br>- Saves fine-tuned model for inference using a pipeline. |
| **Evaluation**            | - Metrics: Accuracy, Macro F1, Micro F1.<br>- Visualizes confusion matrices for both models. |

---

### Model Results

| **Model**                 | **Accuracy** | **Macro F1** | **Micro F1** | **Key Observations**                                                                 |
|---------------------------|--------------|--------------|--------------|-------------------------------------------------------------------------------------|
| **Logistic Regression**   | 0.7950       | 0.7837       | 0.7950       | - Strong performance in specialties like Surgery (F1: 0.86), Obstetrics/Gynecology (F1: 0.95).<br>- Weaker in Consult - History and Phy. (F1: 0.46) due to overlap with other classes.<br>- Benefits from SMOTE and feature selection. |
| **BioBERT (Pipeline)**    | 0.7607       | 0.7538       | 0.7607       | - Competitive performance, especially in Discharge Summary (F1: 0.90), Urology (F1: 0.91).<br>- Struggles with Consult - History and Phy. (F1: 0.43).<br>- Slightly lower than Logistic Regression, possibly due to limited fine-tuning or truncation issues. |

---

### Why Results Are Satisfactory

| **Criteria**              | **Explanation**                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| **Domain Requirements**    | - **Accuracy**: Both models achieve ~76-79% accuracy, acceptable for a multi-class problem with 13 classes and complex medical text.<br>- **F1 Scores**: Macro F1 scores (~0.75-0.78) indicate balanced performance across classes, critical for minority specialties.<br>- **Clinical Relevance**: High F1 scores in critical specialties (e.g., Surgery, Obstetrics) ensure reliability in high-stakes areas. |
| **Industry Standards**     | - Healthcare NLP models typically achieve 70-85% accuracy for text classification due to data complexity.<br>- The project’s results are within this range, suitable for assistive (not standalone) clinical tools.<br>- Confusion matrices show misclassifications are often between similar specialties (e.g., Consult vs. General Medicine), which is less critical than major errors. |
| **Practical Utility**      | - The models reduce manual effort in EHR organization.<br>- BioBERT’s contextual understanding and Logistic Regression’s efficiency make the system versatile for different deployment scenarios. |

---

### Why This Project Is Unique

| **Aspect**                | **Details**                                                                     |
|---------------------------|---------------------------------------------------------------------------------|
| **Hybrid Approach**       | Combines traditional statistical (Logistic Regression) and advanced transformer-based (BioBERT) models, offering flexibility for different computational environments. |
| **Domain-Specific Preprocessing** | Preserves medical keywords and expands abbreviations, ensuring clinical relevance in text processing. |
| **Class Imbalance Handling** | Uses SMOTE and text augmentation to address minority classes, improving model fairness. |
| **BioBERT Fine-Tuning**   | Leverages a domain-specific transformer model (BioBERT), fine-tuned for medical specialty classification, which is rare in open-source projects. |

---

### Strong Points of the Project

| **Strength**              | **Description**                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| **Robust Preprocessing**  | - Custom preprocessing pipeline preserves medical context (e.g., medical keywords, abbreviations).<br>- Handles noisy data effectively (e.g., headers, missing values). |
| **Comprehensive Pipeline** | - End-to-end workflow from data loading to model evaluation.<br>- Includes augmentation, feature selection, and hyperparameter tuning. |
| **Model Diversity**       | - Statistical model (Logistic Regression) is lightweight and interpretable.<br>- BioBERT provides state-of-the-art contextual understanding. |
| **Evaluation**            | - Uses multiple metrics (Accuracy, Macro/Micro F1) and confusion matrices for thorough analysis.<br>- Stratified train-test split ensures balanced evaluation. |
| **Scalability**           | - Saved BioBERT model and pipeline enable easy inference.<br>- Logistic Regression is computationally efficient for large datasets. |

---

### Domain Knowledge

| **Area**                  | **Details**                                                                     |
|---------------------------|---------------------------------------------------------------------------------|
| **Medical Terminology**   | - Incorporates a curated list of medical keywords (e.g., "angioplasty," "biopsy") to preserve during preprocessing.<br>- Expands medical abbreviations (e.g., "bp" to "blood pressure") for clarity. |
| **Healthcare Workflow**   | - Understands the need for specialty classification in EHR systems and clinical decision support.<br>- Addresses class imbalance, reflecting real-world distribution of medical cases. |
| **NLP in Healthcare**     | - Uses BioBERT, a model pre-trained on biomedical texts, ensuring domain-specific embeddings.<br>- Applies TF-IDF with n-grams to capture medical phrases (e.g., "chest xray"). |
| **Clinical Implications** | - Prioritizes high F1 scores for critical specialties to minimize misclassification risks.<br>- Recognizes the importance of balanced performance across classes for equitable healthcare delivery. |

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Vraj-Data-Scientist/medical-text-analysis-for-diagnosis-support
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

## Usage
1. Place `mtsamples.csv` in the project directory.
2. Run the Jupyter notebook `medical-text-analysis-for-diagnosis-support.ipynb`.
3. For inference with BioBERT:
   ```python
   from transformers import pipeline
   classifier = pipeline("text-classification", model="biobert_medical_specialty_classifier", tokenizer="biobert_medical_specialty_classifier")
   result = classifier("Your medical text here", truncation=True, max_length=512)
   print(result)
   ```

## Results
| Model              | Accuracy | Macro F1 | Micro F1 |
|--------------------|----------|----------|----------|
| Logistic Regression| 0.7950   | 0.7837   | 0.7950   |
| BioBERT            | 0.7607   | 0.7538   | 0.7607   |

## Dataset
- **Source**: `mtsamples.csv` (medical transcriptions with specialties).
- **Classes**: 13 medical specialties (e.g., Surgery, Cardiology, others).

## Requirements
- Python 3.8+
- Libraries: pandas, numpy, nltk, scikit-learn, imblearn, transformers, torch, nlpaug, matplotlib, seaborn
- See `requirements.txt` for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
