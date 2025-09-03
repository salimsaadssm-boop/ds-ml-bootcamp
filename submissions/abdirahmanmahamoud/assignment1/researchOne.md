# Research Assignment: Introduction to Machine Learning

### Focus: Business Applications

---

## 1. Defining Machine Learning (with a Business Example)

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables systems to automatically learn and improve from experience without being explicitly programmed (Jordan & Mitchell, 2015). Instead of relying solely on hard-coded rules, ML algorithms analyze data, identify patterns, and make predictions or decisions.

**Real-life Business Example:**
In e-commerce, companies such as Amazon and Alibaba use ML for **product recommendation systems**. For instance, when a customer buys or searches for a product, the algorithm analyzes past purchase history, browsing behavior, and similar customer profiles to recommend other products. This personalized marketing improves sales and customer satisfaction.

---

## 2. Supervised vs. Unsupervised Learning

| Feature              | Supervised Learning                                                                                                       | Unsupervised Learning                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Definition**       | The model is trained on labeled data (input-output pairs).                                                                | The model learns from unlabeled data without predefined outcomes.                                                                             |
| **Goal**             | Predict outcomes based on past examples.                                                                                  | Discover hidden structures or patterns in data.                                                                                               |
| **Business Example** | **Credit Scoring:** Banks use labeled data (approved or rejected loans) to predict if a new applicant is likely to repay. | **Customer Segmentation:** Retailers use unsupervised clustering to group customers by behavior, e.g., frequent buyers vs. seasonal shoppers. |

---

## 3. Overfitting in Machine Learning

**Overfitting** occurs when a model learns training data too well, including noise and irrelevant details, making it perform poorly on new/unseen data (Hastie, Tibshirani & Friedman, 2017).

### Causes:

- Too complex models with too many parameters.
- Insufficient training data.
- Too many training epochs (learning iterations).

### Prevention Techniques:

1. **Cross-validation:** Evaluate the model on multiple folds of the dataset.
2. **Regularization:** Techniques such as L1/L2 regularization to penalize overly complex models.
3. **Dropout & Pruning:** In neural networks, randomly dropping connections during training reduces over-dependence.
4. **More Data:** Expanding the training dataset improves generalization.

---

## 4. Training Data and Test Data Splitting

When building ML models, the dataset is typically split into **training data** (to teach the model) and **test data** (to evaluate performance). Common ratios include **70/30 or 80/20 splits** (Goodfellow et al., 2016).

**Why it’s necessary:**

- Training data helps the model learn patterns.
- Test data evaluates how well the model generalizes to unseen examples.
- Prevents misleadingly high accuracy from models that simply “memorize” training data.

In business, for example, a **fraud detection model** trained on past labeled transactions must be tested on new, unseen transactions to ensure it detects fraud effectively in real life.

---

## 5. Case Study: Machine Learning in Business

**Case Study:** _Predictive Analytics in Retail Supply Chains_ (Waller & Fawcett, 2013).

- **Objective:** Improve inventory management and reduce costs using predictive ML models.
- **Method:** The researchers applied supervised learning models on historical sales data, promotions, and seasonal patterns. They used regression and classification models to forecast product demand.
- **Findings:** Retailers that adopted ML-based predictive analytics saw up to **30% improvement in demand forecasting accuracy**, reduced stockouts, and better customer satisfaction. The study highlighted how ML optimizes logistics and enhances decision-making in supply chains.

---

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT Press.
- Hastie, T., Tibshirani, R., & Friedman, J. (2017). _The Elements of Statistical Learning: Data Mining, Inference, and Prediction_. Springer.
- Jordan, M. I., & Mitchell, T. M. (2015). Machine learning: Trends, perspectives, and prospects. _Science_, 349(6245), 255–260.
- Waller, M. A., & Fawcett, S. E. (2013). Data science, predictive analytics, and big data: A revolution that will transform supply chain design and management. _Journal of Business Logistics_, 34(2), 77–84.
