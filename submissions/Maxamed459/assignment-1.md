# Introduction to Machine Learning — Research Assignment

## What is Machine Learning?

**Machine Learning (ML)** is the science of teaching computers to learn patterns from data and improve over time without being explicitly programmed for every rule.

### Real-life example: Movie Recommendation Systems

Streaming platforms like **Netflix** or **YouTube** use ML to suggest content.

- **Task:** Recommend movies or videos a user is likely to enjoy.
- **Experience:** Past viewing history and user ratings.
- **Performance measure:** Engagement metrics such as watch time or rating accuracy.

The system improves its recommendations the more users interact with it.

---

## Supervised vs. Unsupervised Learning

ML problems are usually grouped into **supervised** or **unsupervised** learning.

| Feature           | Supervised Learning                                               | Unsupervised Learning                                          |
| ----------------- | ----------------------------------------------------------------- | -------------------------------------------------------------- |
| **Goal**          | Learn to map input data to known outputs                          | Discover hidden structure or relationships in data             |
| **Data type**     | Labeled data (inputs + correct answers)                           | Unlabeled data (inputs only)                                   |
| **Typical tasks** | Predicting exam scores, detecting fraud, sentiment analysis       | Market segmentation, topic modeling, anomaly detection         |
| **Example**       | Predicting loan default using applicant details and past outcomes | Grouping students into learning styles based on study behavior |

👉 Supervised learning is like a teacher giving both **questions and answers**.  
👉 Unsupervised learning is like exploring a **puzzle with no guide**, searching for natural groupings.

---

## Overfitting: What It Is and How to Handle It

### Causes of Overfitting

Overfitting happens when a model performs very well on training data but fails on new, unseen data. This occurs when:

- The model is **too complex** for the available data.
- There is **too little data** to learn meaningful patterns.
- The data contains **noise or errors**, which the model memorizes.
- There is **data leakage**, where hidden information about the test set sneaks into training.

### Ways to Prevent Overfitting

- **Simplify the model:** Use fewer parameters or smaller architectures.
- **Regularization:** Apply penalties that discourage overly complex models.
- **Cross-validation:** Test performance on multiple subsets of data.
- **Early stopping:** Halt training once validation performance stops improving.
- **Add more data or augment existing data** to increase variety.

---

## Train/Test Splits: Why They Matter

To evaluate how well a model generalizes, we split data into parts:

- **Training set:** Used to fit the model.
- **Validation set (optional):** Used to fine-tune model settings.
- **Test set:** Used only once at the end, to estimate real-world performance.

### Why this is necessary

If a model is tested on the same data it was trained on, the accuracy score is misleadingly high. Splitting ensures that the model is evaluated on **unseen examples**, mimicking real-world scenarios.

### How splitting is done

- A common ratio is **80% training / 20% testing**.
- For imbalanced problems (e.g., rare diseases), stratified splitting keeps class proportions balanced.
- In time-series problems (e.g., stock forecasting), the split is chronological — training on past data, testing on future data.

---

## Case Study: Machine Learning in Transportation

### Predicting Traffic Congestion with ML

A 2018 study by Lv et al. introduced a **deep learning model** that uses real-time traffic sensor data to predict road congestion in urban areas.

- **Method:** The researchers applied recurrent neural networks (RNNs) to capture time-dependent traffic patterns.
- **Data:** Large-scale traffic flow data collected from city sensors.
- **Findings:** The model achieved high accuracy in forecasting short-term traffic congestion, helping city planners and navigation apps reroute vehicles efficiently.

### Why it matters

This shows how ML can improve **transport efficiency**, reduce travel time, and support **smart city** development.
