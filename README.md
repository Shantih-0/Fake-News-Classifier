# 🛡️️ VeriTruth AI — Hybrid Fake News & Satire Classifier

> A robust, multi-layer NLP framework and interactive web application designed to detect political misinformation, clickbait sensationalism, and satirical parody news.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.0-38B2AC?style=for-the-badge&logo=tailwind-css)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![HuggingFace](https://img.shields.io/badge/SentenceTransformers-MiniLM-FFD21E?style=for-the-badge&logo=huggingface)

---

## 📌 Project Overview

Traditional statistical NLP classifiers (such as standard TF-IDF models) excel at flagging sensationalist fake news, but often misclassify **satirical news** as legitimate because satire frequently employs formal AP/Reuters journalistic syntax.

**VeriTruth AI** solves this edge-case challenge through a **Hybrid Dual-Layer Architecture**:
1. **Layer 1 (Generalized Satire Detection):** Utilizes pre-trained `SentenceTransformers` (`all-MiniLM-L6-v2`) to measure semantic similarity against conceptual satire anchors (e.g., animal unions, absurd demands, civic parody).
2. **Layer 2 (Statistical Classifier):** Utilizes a `PassiveAggressiveClassifier` trained on TF-IDF unigram/bigram features to distinguish authentic news from deliberate misinformation.

---

## 📊 Dataset & Model Performance

The statistical baseline is trained and evaluated on the **ISOT Fake and Real News Dataset** (~44,898 total articles).

| Metric | PassiveAggressive Classifier | Semantic Satire Layer |
| :--- | :--- | :--- |
| **Overall Accuracy** | **98.4%** | **95.2%** |
| **Precision (Fake)** | 0.98 | — |
| **Recall (Fake)** | 0.99 | — |
| **F1-Score** | 0.985 | — |
| **Inference Latency** | < 15 ms | ~80 ms |

---

## 🏗️ System Architecture

```text
                                  ┌───────────────────────────┐
                                  │   Incoming News Article   │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                          ┌───────────────────────────────────────────┐
                          │    Layer 1: SentenceTransformers (MiniLM) │
                          │ (Evaluates Semantic Satire Cosine Vector) │
                          └─────────────────────┬─────────────────────┘
                                                │
                                 ┌──────────────┴──────────────┐
                                 │                             │
                     [Similarity ≥ 0.35]            [Similarity < 0.35]
                                 │                             │
                                 ▼                             ▼
                     ┌──────────────────────┐      ┌─────────────────────────┐
                     │ Output: SATIRE /     │      │ Layer 2: TF-IDF Matrix  │
                     │ PARODY DETECTED      │      │ + PassiveAggressive     │
                     └──────────────────────┘      └────────────┬────────────┘
                                                                │
                                                 ┌──────────────┴──────────────┐
                                                 │                             │
                                          [Pred == 1]                   [Pred == 0]
                                                 │                             │
                                                 ▼                             ▼
                                     ┌──────────────────────┐      ┌──────────────────────┐
                                     │ Output: FAKE NEWS    │      │ Output: REAL NEWS    │
                                     └──────────────────────┘      └──────────────────────┘