# SFHA Capstone: Reddit Mental-Health Topic Classification

A supervised text-classification pipeline that assigns Reddit posts to one of six social–mental health categories addiction, alcoholism, ADHD, bipolar disorder, depression, and divorce using machine-learning workflow from data loading and preprocessing through training, evaluation, and inference.

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Approach and Methodology](#approach-and-methodology)
4. [Performance Metrics](#performance-metrics)
5. [Project Structure](#project-structure)
6. [Prerequisites](#prerequisites)
7. [Installation](#installation)
8. [Data folder](#data-folder-data)
9. [Data source and citation](#data-source-and-citation)
10. [How to Run the Code](#how-to-run-the-code)
11. [Outputs](#outputs)
12. [Limitations and Future Work](#limitations-and-future-work)
13. [Technologies Used](#technologies-used)
14. [License](#license)

---

## Overview

This project implements an end-to-end machine-learning pipeline that classifies Reddit posts into six categories related to mental health and social stressors: **addiction, alcoholism, ADHD, bipolar disorder, depression, and divorce**. It covers data loading, cleaning, text preprocessing, feature extraction, model training, evaluation (including confusion matrices, ROC curves, and EDA plots), and prediction for new text.

---

## Problem Statement

Online platforms such as Reddit have become intimate spaces where people disclose deeply personal struggles with **addiction, alcoholism, ADHD, bipolar disorder, depression, and divorce**. These posts are often written at crisis points: when someone is relapsing into substance use, overwhelmed by untreated ADHD, destabilised by bipolar mood swings, trapped in a depressive episode, or navigating the emotional and practical chaos of marital breakdown. Although these conversations contain rich signals about people’s mental health and support needs, they appear as an unstructured stream of millions of posts distributed across different communities. Human moderators, clinicians, and public-health practitioners cannot manually read or monitor this volume of content in real time.

This project frames the challenge as a **supervised, multi-class text classification problem**. Given a Reddit post (title + body), the model automatically predicts which of six social–mental health categories it most closely reflects: **addiction, alcoholism, ADHD, bipolar disorder, depression, or divorce**. Each category captures a distinct type of psychosocial stressor and risk profile: addiction and alcoholism relate to substance dependence and relapse risk; ADHD to chronic difficulties with attention, impulse control, and functioning; bipolar disorder to severe mood instability; depression to persistent low mood and suicidality risk; and divorce to acute life-change stress, loss, and family disruption. Importantly, **divorce itself is not a mental-health diagnosis** but a major social stressor; it is included deliberately as a contrasting class to help the model distinguish between posts about psychiatric conditions and posts about life events that can trigger or exacerbate those conditions. By training on labelled posts from these topic-specific subreddits, the system learns how people organically describe these problems in their own words, and can then map new, unseen posts to one of these categories at scale.

From a public-health perspective, the classifier turns raw, messy social media narratives into **actionable mental-health intelligence**. Automatically categorising posts in this way makes it possible to:

- Detect **emerging patterns of distress** in near real time (e.g. rising volumes of bipolar or divorce-related posts).
- See **which types of problems are most prominent** in different online communities or time periods.
- Inform **targeted outreach and resource allocation** (for example, whether to prioritise addiction services, depression interventions, or family support programmes).

## Approach and Methodology

The problem is tackled in five main stages.

1. **Data loading and cleaning**  
   All CSVs in the `data/` folder are merged into one DataFrame (`data.py`). Rows with subreddits not in the mental-health list are dropped. Text is cleaned; dates are parsed and filtered to a defined cutoff (e.g. before October 2021). Duplicates are removed by title, snippet, and combined content. A single `content` field is formed by concatenating title and snippet. Categories are balanced to the size of the smallest.

2. **Text preprocessing**  
   Raw text is normalised for modelling: converted to lowercase, stripped of punctuation, tokenised (NLTK), filtered for stopwords (including custom terms such as "I", "im", "hey"), and lemmatised (WordNet). Preprocessing is applied consistently at training and prediction time.

3. **Feature extraction and train–test split**  
   Text is converted to numerical features using TF-IDF (term frequency–inverse document frequency) with configurable n-gram range (default: unigrams and bigrams), document frequency bounds, and maximum feature count. The data is split into train and test sets (e.g. 80% train, 20% test) with a fixed random state for reproducibility.

4. **Model training and selection**  
   Three classifiers are trained on the same TF-IDF features: **Logistic Regression (one-vs-rest)**, **Gradient Boosting**, and **Random Forest**. Cross-validation (or test-set F1 if `--no-cv` is used) is used to estimate performance. The **best model is chosen by weighted F1 score** and is used as the default for saving and for the prediction script.

5. **Evaluation and inference**  
   Each model is evaluated on the held-out test set using classification reports (precision, recall, F1 per class), confusion matrices, and multiclass ROC curves. Optional chi-squared feature analysis highlights terms most associated with each topic. Saved models and the fitted vectoriser can be loaded to classify new text via the `predict.py` script.

---

## Performance Metrics

After training on the balanced dataset of six classes, the pipeline reports metrics such as accuracy, F1-score, and multiclass AUC-ROC for the validation set.

**Accuracy (Logistic Regression – best model):**

- **Validation Accuracy:** **0.81**  
- **Training Accuracy / AUC-ROC:** not reported in this summary (can be obtained from `runner.py` and ROC plots).

**Classification Report (Validation Set – Best Model):**

| Class         | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| addiction    | 0.82      | 0.78   | 0.80     | 546     |
| adhd         | 0.77      | 0.78   | 0.78     | 485     |
| alcoholism   | 0.86      | 0.82   | 0.84     | 535     |
| bipolarreddit| 0.77      | 0.71   | 0.74     | 527     |
| depression   | 0.71      | 0.82   | 0.76     | 503     |
| divorce      | 0.91      | 0.92   | 0.91     | 540     |

Overall validation accuracy (macro and weighted averages) is approximately **0.81**, as reported by the Logistic Regression classification report.

---

## Project Structure

```
SFHA-Capstone/
├── config.py              # Paths, TF-IDF and model hyperparameters, mental-health subreddit list
├── data.py                # Load/merge CSVs from data/, clean text, balance categories
├── preprocess.py          # NLTK preprocessing: tokenise, stopwords, lemmatise
├── model.py               # TF-IDF, train/test split, 3 classifiers, best by F1
├── evaluate.py            # Classification reports, confusion matrices, ROC, chi2 features
├── predict.py             # Load pipeline and metadata; predict topic for new text
├── eda.py                 # EDA plots: subreddit counts, post-length distributions
├── runner.py              # Main: load → preprocess → train → evaluate
├── fetch_subreddit_data.py # Optional: fetch Reddit via PRAW; writes CSV into data/
├── requirements.txt
├── data/                  # CSV files (merged into one DataFrame by data.py)
├── output/                # Plots (confusion matrices, EDA)
├── models/                # Saved vectoriser, classifiers, best_model_name.txt, metadata.json
└── notebooks/             # Optional: reference notebook
```

| File or folder | Purpose |
|----------------|--------|
| `config.py` | Paths, TF-IDF and model settings, `MENTAL_HEALTH_SUBREDDITS`, `SUBREDDIT_TO_LABEL` |
| `data.py` | Merge all CSVs in `data/`, clean text, map to mental-health labels, balance categories |
| `preprocess.py` | Text preprocessing (NLTK; downloads `punkt_tab`/`punkt`, stopwords, wordnet if missing) |
| `model.py` | TF-IDF, train/test split, Logistic Regression / Gradient Boosting / Random Forest; best by F1 |
| `evaluate.py` | Classification reports, confusion matrices, ROC curves, chi2 top features |
| `predict.py` | Load vectoriser, model, and metadata; predict topic (single or batch) |
| `eda.py` | EDA: subreddit counts, post-length histograms and boxplot |
| `runner.py` | Main entry; loads from `data/` by default |
| `fetch_subreddit_data.py` | Optional: fetch Reddit via PRAW; writes to `data/reddit_corpus.csv` |
| `data/` | CSV data (per-subreddit or single file); all merged by `data.py` |
| `output/` | Generated figures |
| `models/` | Saved pipeline and `metadata.json` (topic_codes, topic_labels) |

---

## Prerequisites

- **Python**: 3.8 or higher recommended  
- **Operating system**: Any supported by Python (Windows, macOS, Linux)

---

## Installation

1. **Clone or download the project**  
   Ensure all project files are in a single directory (e.g. `SFHA-Capstone`).

2. **Create and activate a virtual environment (recommended)**  
   - Windows (PowerShell):  
     `python -m venv venv` then `.\venv\Scripts\Activate.ps1`  
   - macOS/Linux:  
     `python3 -m venv venv` then `source venv/bin/activate`

3. **Install dependencies**  
   From the project root directory, run:
   ```bash
   # Windows (PowerShell)
   python -m pip install -r requirements.txt

   # macOS / Linux
   pip install -r requirements.txt
   ```
   This installs pandas, numpy, scikit-learn, NLTK, matplotlib, seaborn, joblib, wordcloud, requests, and praw.

4. **Download NLTK data (if not already present)**  
   The first run will try to download NLTK resources automatically. If you see `LookupError: Resource 'punkt_tab' not found`, run:
   ```bash
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

---

## Data folder (`data/`)

All CSV data lives in the **`data/`** folder. **`data.py`** merges every CSV in this folder into one DataFrame, then cleans text, maps subreddits to mental-health labels, and balances categories.

- **Place CSV files in `data/`.** Each file must have at least the first four columns: **subreddit**, **author**, **date**, **post** (or equivalent). Only subreddits in `config.MENTAL_HEALTH_SUBREDDITS` are kept; others are dropped. Categories are balanced to the size of the smallest.
- **Optional – Fetch from Reddit:** Set `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`, then run `python fetch_subreddit_data.py`. This writes **`data/reddit_corpus.csv`**, which is merged with any other CSVs when you run the pipeline.

Override the data folder with the environment variable `DATA_FOLDER` or `runner.py --data-folder PATH`.

---

## Data source and citation

The original subreddit-level dataset and mental-health support group definitions are based on:

> **Low, D. M., Rumker, L., Torous, J., Cecchi, G., Ghosh, S. S., & Talkar, T. (2020).  
> Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study.  
> *Journal of Medical Internet Research, 22*(10), e22635.**

BibTeX:

```text
@article{low2020natural,
  title={Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study},
  author={Low, Daniel M and Rumker, Laurie and Torous, John and Cecchi, Guillermo and Ghosh, Satrajit S and Talkar, Tanya},
  journal={Journal of medical Internet research},
  volume={22},
  number={10},
  pages={e22635},
  year={2020},
  publisher={JMIR Publications Inc., Toronto, Canada}
}
```

This implementation adapts and extends their Reddit mental-health mapping approach into a reusable, end-to-end classification pipeline.

---

## How to Run the Code

All commands are run from the **project root** (the folder that contains `runner.py` and `data/`).

### 1. Put your data in `data/`

Place one or more CSV files in `data/` (e.g. per-subreddit files or a single corpus). `data.py` merges them all when the pipeline runs.

### 2. Install dependencies and NLTK data (one-time)

```powershell
# From project root (e.g. SFHA-Capstone)
py -m pip install -r requirements.txt
py -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 3. Run the pipeline

**Using the folder of CSVs in `data/` (default):**
```powershell
py runner.py
# or explicitly:
py runner.py --data-folder
```
If `data/reddit_corpus.csv` is missing, the script automatically loads from the `data/` folder.

**Faster run (no cross-validation), with EDA and saving the best model:**
```powershell
py runner.py --no-cv --eda --save-model --chi2
```

**Useful flags:**  
- `--data-folder` – Load from `data/` as a folder of CSVs (first 4 columns).  
- `--no-cv` – Skip cross-validation (faster).  
- `--eda` – Save EDA plots to `output/`.  
- `--save-model` – Save vectoriser and models to `models/` (needed for `predict.py`).  
- `--chi2` – Print top chi-squared features per class.

### 4. Predict with a saved model (after running with `--save-model`)

```powershell
py predict.py --text "I have been feeling very anxious lately"
```

To use a specific model: `py predict.py --model lgreg --text "..."`  
Allowed `--model`: `lgreg`, `gradient_boosting`, `random_forest`.

### Optional: EDA only

```powershell
py data.py
py eda.py
```

(Or run `py eda.py` alone; it loads the corpus from `data/` via `data.py`.)

---

## Outputs

- **Console:**  
  - Row count and subreddit counts after loading.  
  - Cross-validation metrics or test-set F1 scores (depending on flags).  
  - Best model name and its F1.  
  - Per-model classification reports and, if requested, chi-squared feature lists.

- **`output/`:**  
  - Confusion matrix plots per model (e.g. `lgreg_confusion.png`, `gradient_boosting_confusion.png`, `random_forest_confusion.png`).  
  - ROC curves per model (e.g. `lgreg_roc.png`, `gradient_boosting_roc.png`, `random_forest_roc.png`).  
  - If `--eda` was used: `counts.png`, `post_length_overall.png`, `post_length_boxplot.png`, `post_length_violin.png`, `wordcloud.png`.

- **`models/` (when using `--save-model`):**  
  - `vectorizer.joblib` – Fitted TF-IDF vectoriser.  
  - `lgreg_model.joblib`, `gradient_boosting_model.joblib`, `random_forest_model.joblib` – Sklearn classifiers.  
  - `best_model_name.txt` – Name of the best model (used by `predict.py` by default).

---

## Limitations and Future Work

- **Label quality and scope**  
  - Subreddit membership is only a *proxy* for a person’s true mental-health status. People post in communities that do not always match their formal diagnoses, and many experience multiple overlapping problems (e.g. addiction and depression) even though the model predicts a single dominant label.  
  - The current project focuses on six categories (addiction, alcoholism, ADHD, bipolar disorder, depression, divorce). This is useful for demonstration and analysis but does not cover the full spectrum of mental-health and social issues discussed online.

- **Divorce as a social stressor, not a diagnosis**  
  - Divorce is included as a **social problem class**, not as a mental-health condition. Its role is to capture posts about relationship breakdown and life disruption and to provide contrast against clinical categories. The model’s outputs must not be interpreted as clinical judgments about individuals’ psychiatric status.

- **Data and population bias**  
  - The training data comes from Reddit, which skews toward particular age groups, cultures, and levels of digital access. Results cannot be assumed to generalise to the whole population or to offline settings.  
  - All analysis is based on English-language posts; expressions of distress in other languages and cultures may look very different.

- **Model class and feature limitations**  
  - The current pipeline uses TF-IDF features and classical machine-learning models (Logistic Regression, Gradient Boosting, Random Forest). These capture word- and phrase-level patterns but do not model deeper context, sarcasm, or long conversational history as well as modern transformer-based models.  
  - The model operates at the **single-post** level and does not use longitudinal information (e.g. how a user’s posts change over time) or network structure (who interacts with whom), both of which are important for understanding risk and resilience.

- **Ethical and practical constraints**  
  - The system is designed for **aggregate, population-level insights**, not to flag or intervene on individuals. Any deployment that attempts to act at the individual level would need additional safeguards, consent processes, and clinical oversight.  
  - Models can drift as language and subreddit culture change. Without periodic retraining and monitoring, performance and fairness may degrade over time.

**Future improvements** could include:

- Incorporating **contextual language models** (e.g. transformers) to better capture nuance, co-occurring themes, and implicit emotional states.
- Expanding or refining labels to distinguish comorbid or adjacent issues (e.g. anxiety, trauma, financial stress) and to support multi-label classification when posts clearly span multiple problem domains.
- Integrating **temporal** features (how topics and risk signals evolve over weeks or months) and basic network information to identify early warning patterns more reliably.
- Performing systematic **fairness and bias audits** across demographic groups and communities, and adjusting data and models to mitigate observed disparities.
- Working with clinicians, peer-support organisations, and public-health teams to design **interpretation guidelines and dashboards** that make the outputs useful for real-world decision-making while respecting privacy and ethical boundaries.

---

## Technologies Used

- **Language:** Python 3  
- **Data and numerics:** pandas, NumPy  
- **NLP:** NLTK (tokenisation, stopwords, lemmatisation)  
- **Machine learning:** scikit-learn (TF-IDF, train/test split, cross-validation, Logistic Regression, Gradient Boosting, Random Forest, metrics, chi2)  
- **Visualisation:** matplotlib, seaborn, wordcloud  
- **Persistence:** joblib (models and vectoriser)  
- **Utilities:** requests, praw, standard library (argparse, os)

---

## License

See the `LICENSE` file in the repository.
