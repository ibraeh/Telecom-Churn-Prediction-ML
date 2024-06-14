### Project Outline

**Title**: 
[Telecom Customer Churn Prediction using Machine Learning Algorithms](https://www.kaggle.com/code/bigibraeh/customer-churn-prediction)  
**Duration**: 2 Months  
**Team Size**: 1

**Objective**: 
To develop a model for predicting customer churn in the telecom industry, aiming to provide insights for proactive customer retention strategies.

**Scope**: 
Predicting customer churn is crucial for telecom companies to minimize loss of revenue and maintain customer satisfaction. This project evaluated multiple machine learning algorithms to identify the best approach for accurate churn prediction.

**Role and Responsibilities**:  
- **Role**: Machine Learning Engineer
- **Responsibilities**:
  - Collected and preprocessed data from Kaggle, including cleaning and encoding categorical features.
  - Conducted Exploratory Data Analysis (EDA) to understand data distribution and characteristics.
  - Applied feature selection techniques such as Variance Threshold, ANOVA, Recursive Forward Elimination, Random Forest Feature Importance, and PCA.
  - Developed models using various algorithms, including Logistic Regression, Linear Discriminant Analysis, Naive Bayes, Decision Trees Classifier, K-Nearest Neighbor, Multi-Layer Perceptron, Random Forest, and XGBoost.
  - Evaluated model performance using metrics such as accuracy, precision, recall, and AUC-ROC.
  - Fine-tuned hyperparameters using GridSearchCV to optimize model performance.
  - Used SHAP (SHapley Additive exPlanations) for model interpretation and to identify key features influencing churn.

**Methods and Tools**:  
- **Data Sources**:  [Telecom Customer Churn Prediction](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics) dataset with customer details including demographics, subscription services, and churn status.
- **Algorithms**: Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbor, Decision Tree Classifier, Naïve Bayes, Multi-Layer Perceptron, XGBoost, Random Forest.
- **Tools and Libraries**: Jupyter Notebook, Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, SHAP.

**Challenges and Solutions**:  
- **Challenges**:
  - Imbalanced data with fewer churn cases compared to non-churn cases.
  - High-dimensional categorical features requiring effective encoding and selection.
- **Solutions**:
  - Applied SMOTE for oversampling to balance the dataset.
  - Used label encoding and one-hot encoding for categorical features.
  - Employed feature selection techniques to identify the most relevant features and reduce dimensionality.

**Results and Impact**:  
- **Key Achievements**:
  - Developed a predictive model with 90% accuracy in predicting customer churn. 
  - The XGBoost model outperformed and made improvement from the baseline logistic regression model with more than 25%.
  - XGBoost also performed well in terms of precision, recall, and F1 score across all classes.
- **Metrics**: 
  - XGBoost achieved F1 score of 0.92 and accuracy of 0.92
  - Precision and Recall were as follows 0.92 and 0.92. Consistently high across all models, with XGBoost leading.
  - AUC-ROC score of XGBoost in the multi-class were 0.96, 1.00, and 0.98 for Churned, Joined and Stayed respectively.
- **Impact**: 
  - The predictive model provided telecom companies with actionable insights for customer retention, reducing churn rates and improving customer satisfaction.

**Conclusion**:  
- **Lessons Learned**: 
  - Ensemble algorithms, particularly XGBoost, are highly effective for churn prediction.
  - Feature selection using techniques like Random Forest Feature Importance is crucial for model interpretability and performance.
- **Future Work**: 
  - Explore advanced deep learning techniques such as recurrent neural networks.
  - Address challenges related to computational complexity and model interpretability.

 

[View Notebook](https://www.kaggle.com/code/bigibraeh/customer-churn-prediction)
* Screenshots

![model_scores](https://github.com/ibraeh/Telecom-Churn-Prediction-ML/assets/29314702/213e83e8-19b4-45c5-a91a-fb31abace5f7)
![SHAP summary plot](https://github.com/ibraeh/Telecom-Churn-Prediction-ML/assets/29314702/cc07f4c7-dfc4-4138-a309-2682585b98a5)
![SHAP feature importances](https://github.com/ibraeh/Telecom-Churn-Prediction-ML/assets/29314702/fb686a26-53a5-4029-b184-8d9dbe8d0b0f)
