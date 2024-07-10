# Loan-Prediction-Analysis

## Description
The Loan Prediction Analysis project uses machine learning to predict loan approval for applicants based on their personal and financial details. It leverages historical data to help financial institutions make informed decisions efficiently and accurately.

## Technologies Used
- **Python**: Programming language for data analysis and modeling.
- **Pandas**: Data manipulation and preprocessing.
- **Scikit-Learn**: Machine learning library, using Logistic Regression.
- **MongoDB**: NoSQL database for data storage.
- **PyMongo**: Library for MongoDB interaction.

## Purpose
To automate and enhance the loan approval process by providing:
- **Efficiency**: Faster loan assessments.
- **Accuracy**: Data-driven decisions.
- **Consistency**: Uniform evaluation criteria.
- **Scalability**: Handles large datasets.

## Workflow
1. **Data Loading**: From MongoDB.
2. **Data Preprocessing**: Cleaning and encoding.
3. **Model Training**: Logistic regression.
4. **Model Evaluation**: Accuracy assessment.
5. **Prediction for New Applicants**: Interactive loan status prediction.
6. **Results Storage**: In MongoDB for future use.

## Example Usage
To predict loan status for a new applicant, use the `predict_loan_status` function which collects user input interactively:

```python
predicted_status = predict_loan_status()
print(f'Predicted Loan Status: {predicted_status}')
