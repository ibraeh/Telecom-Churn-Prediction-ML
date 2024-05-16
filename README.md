# Project Title
[Telecom Customer Churn Prediction using Machine Learning Algorithms]
(https://www.kaggle.com/code/bigibraeh/customer-churn-prediction)

# Description 
Customer churn is a critical concern for telecom industry, where customers can easily switch providers. Predicting churn is vital for telecom business strategy. This project aims to evaluate the effectiveness of various machine learning algorithms in predicting customer churn.

This project investigates the following questions:

* What factors contribute to customer churn in the telecom industry?
* How accurately can different machine learning algorithms predict customer churn based on these factors?

# Methodology
## Data Acquisition
The dataset was obtained from Kaggle using the link [Telecom Customer Churn Prediction](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics). The dataset contained customer details such as demographics, location, tenure, subscription services, customer status (joined, stayed, or churned), and many more.

## Data Exploration & Preprocessing
* Exploratory Data Analysis (EDA): Initial exploration was performed on the data to understand its characteristics, identify missing values, and analyze key features distribution.
* Data Cleaning: Missing values were removed due to the features being categorical in nature. Techniques like imputation were not used since this worked best for numerical values. The categorical features were also encoded using appropriate methods such as label encoding and one-hot encoding. The features were scaled using standardization and the presence of outliers were mitigated using this approach. Again, the imbalance nature of the dataset was handled through the application SMOTE and Random Under Sampler.


## Machine Learning Models
A mix of linear, non-linear and ensemble algorithms were used for this multi-class classification problem. Below is the breakdown:

* Linear algorithms: Logistic regression and Linear Discriminant Analysis.
* Non-linear algorithms: K-Nearest Neighbor, Decision Tree Classifier, Naïve Bayes, and Multi-Layer Perceptron. 
* Ensemble algorithms: Extreme Gradient Boosting and  Random Forest.

# Evaluation
To assess the effectiveness of the chosen models in predicting customer churn, the following evaluation metrics were employed: Accuracy, Precision, Recall, and AUC-ROC Curve.

Each model was evaluated using these metrics on a GridSearchCV to determine the best and optimize model. The results were presented in a table.

**Key Findings**:

Based on the GridSearchCV, the most effective model for predicting customer churn was determined. The analysis also highlighted on:

* Which model achieved the best score, accuracy, precision, recall, and AUC-ROC score.
* Insights into the trade-offs between different metrics.

# Key Takeaways

* Effectiveness of Machine Learning Algorithms: Ensemble algorithms like XGBoost were effective in predicting customer churn with an accuracy of more than 90%. It was also consistently higher across all the classes in terms of precision, recall, and F1 score showing that it was a robust and scalable model. Compared to the baseline model with an accuracy of 70%, XGBoost improved in more than 25%.  
* Impactful Features: The preferred feature selection technique out of the five was Random Forest Feature Importance. This was due to its effectiveness, interpretability, and robustness in dealing with complexity in data. Which was valuable in understanding customer behavior in the telecom industry.  
* Limitations & Considerations: SVM was not used due to its computational and time complexity. It will be worth exploring and applying deep learning in future works. Some challenges that could also be addressed in future endeavors include limitations in the size and nature of the dataset, complexity of the problem, computational resource availability, and interpretability of the model.

**Conclusion**: Overall, XGBoost was effective in predicting customer churn and ideal strategy in enhancing customer retention in the telecom industry. The project also underscored the importance of selecting the appropriate feature selection technique and models tailored to a task. 


