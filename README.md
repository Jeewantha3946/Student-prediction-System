# Student Performance Prediction System

## Project Overview

This project is a machine learning app designed to estimate a student's exam score based on factors like study habits, attendance, past marks, and home support. It covers the full machine learning process—from cleaning the data and training models to comparing results and deploying a working Streamlit app.

The goal was to make the tool interactive and easy to use. Users can enter a student's details, get an estimated exam score, view model performance charts, and download their input data along with the predicted score as a CSV file.

## Dataset

Dataset: Student Performance Factors

Total records before cleaning: 6,607

Total records after cleaning: 6,606

Number of features: 19

Target variable: Exam_Score

Score range in cleaned data: 55 to 100

The dataset contains both numerical and categorical information about students, with Exam_Score being the target value our models try to predict.

## Data Preparation

We found one record with an invalid score of 101, which was removed before training. This left us with 6,606 clean rows of data.

We split the cleaned data into an 80% training set and a 20% test set. To keep everything consistent between training and testing, we built a scikit-learn pipeline to handle all preprocessing:

Missing numbers were filled using the median value.

Numerical features were standardized with StandardScaler.

Missing categories were filled using the most frequent value.

Categorical text was converted into numbers using one-hot encoding.

Using a pipeline prevents data leakage and ensures the app handles new user inputs the exact same way during live predictions.

## Machine Learning Models

We tested several regression models:

Dummy Regressor (baseline)

Linear Regression

Random Forest

Tuned Random Forest

XGBoost

Tuned XGBoost

### Model Selection

Linear Regression performed best during 5-fold cross-validation, making it our final choice. Its average cross-validation results were:MAE: 0.5116RMSE: 2.0840$R^2$: 0.7166Linear Regression achieved the lowest error rates (MAE and RMSE) and the highest score ($R^2$) among all tested models. Tuned XGBoost came close, but it couldn't beat Linear Regression on this dataset. We saved the final Linear Regression pipeline as models/best_model.pkl to power the Streamlit app.Note: Cross-validation score shows how the model performed across different validation splits during training, while the final test score reflects how it handled completely unseen test data at the end.

## Final Test Performance

| Metric | Value |
|---|---:|
| MAE | 0.4160 |
| RMSE | 1.5213 |
| R² | 0.8250 |

The test results show that the model performs well overall. The MAE of 0.4160 means that, on average, the predicted score is off by less than half a point. The $R^2$ score of 0.8250 means the model explains around 82.5% of the variation in exam scores.The RMSE is slightly higher than the MAE because it penalizes larger errors more heavily. This indicates that while most predictions are very accurate, there are a few outlier students where the prediction was less precise.

## Technologies

- Python 3.13
- Scikit-learn 1.9.0
- Streamlit 1.66.1
- Pandas
- NumPy
- Matplotlib
- Joblib
- Jupyter Notebook

## Project Structure

Student_Performance_Prediction/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── Student_Performance_Prediction.ipynb
├── StudentPerformanceFactors.csv
├── data/
│   └── student_performance_cleaned.csv
├── models/
│   └── best_model.pkl
└── outputs/
    ├── actual_vs_predicted.png
    ├── error_distribution.png
    ├── exam_score_distribution.png
    ├── exam_score_boxplot.png
    ├── feature_importance.png
    ├── final_model_comparison.png
    ├── permutation_importance.png
    ├── residual_analysis.png
    └── Other evaluation files

## Installation

From the project directory, install the required packages:


python -m pip install -r requirements.txt


## Run the Application

Start the Streamlit application with:


python -m streamlit run app.py


Then open the local address shown by Streamlit, normally:

<http://localhost:8501>

## Application Features

The application provides:

- A dashboard showing the selected model and final test metrics
- A student performance prediction form
- Input fields for all 19 student features
- A predicted exam score
- Model comparison and evaluation information
- Model insights and visualisations
- Input validation
- CSV export for an individual prediction

## Model Limitations

The model is less reliable for some high-scoring students. The cleaned
dataset contains many more observations in the middle score ranges than at
the very top of the range; the 81–100 score group contains only a small number
of test observations. With limited examples of very high performance, a
regression model tends to predict closer to the average and may underestimate
some exceptional scores.

The model also learns patterns from the available dataset rather than proving
that any particular factor causes a student's result. Prediction quality may
change for students from different schools, regions, or populations, or when
the input data is outside the patterns seen during training. The test metrics
are therefore not a promise of equal accuracy for every student.

This application should be used for educational exploration and as a
supporting analytical tool only. It should not be used on its own for
high-stakes decisions such as grading, admissions, student discipline,
funding, or access to educational opportunities.

## Live Application

Streamlit URL: *Add the deployed application URL here after deployment.*

## Academic Integrity

AI tools were used to support the wording and presentation of this README.
The dataset preparation, model development, evaluation, and final project
decisions remain the responsibility of the project author. The author should
update this declaration if any other AI tools were used for coding, analysis,
or content generation, so that it accurately reflects the work submitted.
