# Heart Disease Prediction using CatBoost

## Overview

This project aims to predict the likelihood of heart disease using machine learning, specifically the CatBoost algorithm. It utilizes the Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset, focusing on key health indicators.

## Dataset

The dataset used is `heart_disease_health_indicators_BRFSS2015.csv` from the BRFSS 2015 survey. It includes various health-related features such as BMI, smoking habits, blood pressure, and physical activity.

## Methodology

1. **Data Collection and Preprocessing:**
   - Load the dataset using pandas.
   - Handle missing values and outliers.
   - Encode categorical features.
   - Scale numerical features using StandardScaler.

2. **Exploratory Data Analysis (EDA):**
   - Analyze data distributions and correlations using visualizations like box plots, histograms, and heatmaps.
   - Identify potential predictors and understand relationships between features.

3. **Feature Engineering:**
   - Select relevant features based on domain knowledge and EDA insights.
   - Consider creating new features if necessary.

4. **Model Selection and Training:**
   - Choose CatBoost as the classification algorithm due to its performance and handling of categorical features.
   - Split the data into training and testing sets.
   - Address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
   - Train the CatBoost model on the resampled training data.

5. **Hyperparameter Tuning:**
   - Optimize model performance using GridSearchCV to find the best hyperparameter settings.
   - Evaluate different combinations of hyperparameters based on cross-validation accuracy.

6. **Model Evaluation:**
   - Evaluate the best model on the test set using metrics like accuracy, precision, recall, and F1-score.
   - Visualize the results using a confusion matrix.

## Results

- The model achieved [insert accuracy score] accuracy on the test set.
- [Mention any other relevant evaluation metrics and insights].

## Conclusion

This project demonstrates the effectiveness of using CatBoost for heart disease prediction. The selected features and tuned hyperparameters contribute to the model's performance. Further improvements could be explored by incorporating more advanced feature engineering techniques and experimenting with other algorithms.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- imblearn
- catboost
- matplotlib
- seaborn

## Usage

1. Upload the dataset (`heart_disease_health_indicators_BRFSS2015.csv`) to your Google Colab environment.
2. Run the notebook cells sequentially to execute the code.
3. Modify or extend the code as needed for your specific requirements.
