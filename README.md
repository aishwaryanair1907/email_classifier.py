# email_classifier


# Email Classifier

A simple Python-based email classifier that categorizes emails into three categories: spam, malicious, and genuine. This classifier uses Natural Language Processing (NLP) techniques along with a Naive Bayes machine learning model to classify emails based on their content.

## Features

- Classifies emails as `spam`, `malicious`, or `genuine`.
- Uses TF-IDF for text vectorization.
- Naive Bayes classifier for efficient text classification.
- Evaluates model performance using accuracy and classification report metrics.

## Requirements

- Python 3.6 or higher
- `pandas`
- `scikit-learn`
- `nltk` (if using additional preprocessing like stopword removal)
- `numpy`

You can install the required packages using pip:

bash
pip install pandas scikit-learn nltk numpy


## Dataset

The dataset for training and testing consists of sample emails labeled as `spam`, `malicious`, or `genuine`. The dataset is defined directly within the Python script for simplicity. In a real-world scenario, you would replace this with a larger, more diverse dataset.

## Usage

1. **Clone the Repository**: 

   Clone this repository to your local machine or download the `email_classifier.py` script.

2. **Run the Classifier**:

   bash
   python email_classifier.py
   

   This will train the classifier on the predefined dataset, evaluate its performance on a test set, and classify a few new sample emails.

3. **Output**:

   - The script will display the accuracy and classification report, showing precision, recall, and F1-score for each category.
   - It will also show predictions for new sample emails.

## Code Explanation

- **Dataset Preparation**: The dataset is defined as a dictionary containing sample emails and their corresponding labels.
- **Data Splitting**: The data is split into training and testing sets using `train_test_split` with stratification to ensure balanced class representation.
- **Text Vectorization**: TF-IDF vectorization is used to convert text data into numerical format suitable for machine learning models.
- **Model Training**: A Naive Bayes classifier (`MultinomialNB`) is trained on the vectorized text data.
- **Model Evaluation**: The model's performance is evaluated using accuracy and a detailed classification report.
- **Prediction**: The trained model is used to predict the category of new email samples.

## Example Output

plaintext
Accuracy: 0.67

Classification Report:
               precision    recall  f1-score   support

     genuine       1.00      1.00      1.00       2.0
   malicious       0.67      1.00      0.80       2.0
        spam       1.00      0.50      0.67       2.0

    accuracy                           0.83       6.0
   macro avg       0.89      0.83      0.82       6.0
weighted avg       0.89      0.83      0.82       6.0


New Email Predictions:
Email: You have won a lottery! Click here to claim your prize!
Predicted Label: spam

Email: Urgent: Your account has been compromised, please change your password immediately.
Predicted Label: malicious

Email: Hi, let's catch up this weekend over coffee.
Predicted Label: genuine


## Limitations

- **Small Dataset**: The example uses a small, predefined dataset. For more robust performance, a larger and more diverse dataset should be used.
- **Simple Model**: The Naive Bayes classifier is a good starting point but may not be as effective as more sophisticated models for complex data.
- **No Advanced Preprocessing**: Basic text cleaning is applied. Further steps like stemming, lemmatization, or handling special characters could improve performance.

## Future Improvements

- Integrate a larger and more diverse dataset.
- Implement advanced preprocessing techniques (e.g., stopword removal, stemming, lemmatization).
- Experiment with other classifiers (e.g., SVM, Random Forest, Deep Learning models).
- Deploy the classifier as a web service using Flask or Django for real-time email classification.

## License

This project is open source and available under the MIT License.

