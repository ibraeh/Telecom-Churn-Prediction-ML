# Project Title
[Telecom Customer Churn Prediction using Machine Learning Algorithms](https://www.kaggle.com/code/bigibraeh/customer-churn-prediction)

# Description 
Customer churn is a critical concern for the telecom industry, where customers can easily switch providers. Predicting churn is vital for telecom business strategy. This project aims to evaluate the effectiveness of various machine learning algorithms in predicting customer churn.

This project investigates the following questions:

* What factors contribute to customer churn in the telecom industry?
* How accurately can different machine learning algorithms predict customer churn based on these factors?

# Methodology
## Data Acquisition
The dataset was obtained from Kaggle using the link [Telecom Customer Churn Prediction](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics). The dataset contains customer details such as demographics, location, tenure, subscription services, customer status (joined, stayed, or churned), and more.

## Data Exploration & Preprocessing
* **Exploratory Data Analysis (EDA):** Initial exploration was performed on the data to understand its characteristics, identify missing values, and analyze key feature distributions.
* **Data Cleaning:** Missing values were removed due to the features being categorical. Techniques like imputation were not used since this works best for numerical values. The categorical features were also encoded using appropriate methods such as label encoding and one-hot encoding. The features were scaled using standardization, and outliers were mitigated using this approach. Again, due to the imbalance in the dataset, oversampling and undersampling were performed using SMOTE and Random Under Sampler, respectively.
* **Feature Selection:** Five different feature selection techniques were used including Variance threshold, ANOVA with select K best, Recursive Forward Elimination, Random Forest Feature Importance, and Principal Component Analysis (PCA).


## Machine Learning Models
A mix of linear, non-linear, and ensemble algorithms were used for this multi-class classification problem. Below is the breakdown:

**Linear algorithms:** 
* Logistic regression: Used as the baseline model, interpretable, and efficient.  
* Linear Discriminant Analysis: Handles high dimensionality, good for separating classes if the data is normally distributed.
 
**Non-linear algorithms:** 
* K-Nearest Neighbor: Non-parametric, efficient in capturing local patterns.
*  Decision Tree Classifier: Handles mixed features, interpretable, good for feature importance.
* Naïve Bayes: Efficient for large datasets, works well with sparse features (if feature independence assumption holds).
* Multi-Layer Perceptron: Explores non-linear relationships, and learns complex patterns.
  
**Ensemble algorithms:** 
* Extreme Gradient Boosting (XGBoost): Robust, handles complex data, effective in classification tasks, provides feature importance.
* Random Forest: Handles non-linearity through bagging, reduces overfitting, interpretable with feature importance.

## Tools
Jupyter Notebook, Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, SHAP.

# Evaluation
To assess the effectiveness of the chosen models in predicting customer churn, the following evaluation metrics were employed: Accuracy, Precision, Recall, and AUC-ROC Curve.

Each model was evaluated using these metrics with GridSearchCV and hyperparameter tuning to determine the best and most optimized model. The results were presented in a table.

**Key Findings**:

Based on the GridSearchCV, the most effective model for predicting customer churn was determined. The analysis also highlighted:

* Which model achieved the best score, accuracy, precision, recall, and AUC-ROC score?
* Insights into the trade-offs between different metrics.

# Key Takeaways

* **Effectiveness of Machine Learning Algorithms:** Ensemble algorithms like XGBoost were effective in predicting customer churn with an accuracy of 90%. It was also consistently higher across all the classes in terms of precision, recall, and F1 score showing that it was a robust and scalable model. Compared to the baseline model with an accuracy of 70%, XGBoost improved by more than 20%.  
* **Impactful Features:** The preferred feature selection technique out of the five was Random Forest Feature Importance. This was due to its effectiveness, interpretability, and robustness in dealing with complexity in data, which was valuable in understanding customer behaviour in the telecom industry.  
* **Limitations & Considerations:** SVM was not used due to its computational and time complexity. It will be worth exploring and applying deep learning especially recurrent neural networks in future works. Some challenges that could also be addressed in future endeavors include limitations in the size and nature of the dataset, complexity of the problem, computational resource availability, and interpretability of the model.

**Conclusion**: Overall, XGBoost was effective in predicting customer churn, and using SHAP (SHapley Additive exPlanations) to explain influential features provided valuable insights in enhancing customer retention strategies in the telecom industry. The project also underscored the importance of selecting the appropriate feature selection technique and models tailored to the task. 

[View Notebook](https://www.kaggle.com/code/bigibraeh/customer-churn-prediction)

* Screenshots
![model_scores](https://github.com/ibraeh/Telecom-Churn-Prediction-ML/assets/29314702/213e83e8-19b4-45c5-a91a-fb31abace5f7)
![SHAP summary plot](https://github.com/ibraeh/Telecom-Churn-Prediction-ML/assets/29314702/cc07f4c7-dfc4-4138-a309-2682585b98a5)
![SHAP feature importances](https://github.com/ibraeh/Telecom-Churn-Prediction-ML/assets/29314702/fb686a26-53a5-4029-b184-8d9dbe8d0b0f)
