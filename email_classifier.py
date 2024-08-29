import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset with more examples
data = {
    'email_text': [
        "Congratulations, you've won a free iPhone! Click here to claim.",
        "Your account has been compromised. Download this software to secure your account.",
        "Dear customer, your invoice is attached. Please review it at your earliest convenience.",
        "Get a free vacation now! Limited offer, click to redeem.",
        "Alert: Suspicious activity detected on your account. Log in to verify.",
        "Meeting reminder: Tomorrow at 10 AM.",
        "Claim your free coupon now by clicking this link.",
        "Important security update: Change your password immediately.",
        "Happy birthday! Hope you have a great day!",
        "Final warning! Pay your overdue bill now to avoid service interruption.",
        "This is not a drill. Please update your security settings immediately.",
        "Your friend sent you a gift! Open now.",
        "Urgent: Your bank account has been accessed from an unknown device.",
        "Your account balance is low. Please add funds to avoid penalties.",
        "You've received a new voice message. Click to listen.",
        "Your subscription is expiring soon. Renew now to avoid disruption."
    ],
    'label': [
        "spam",
        "malicious",
        "genuine",
        "spam",
        "malicious",
        "genuine",
        "spam",
        "malicious",
        "genuine",
        "spam",
        "malicious",
        "spam",
        "malicious",
        "genuine",
        "spam",
        "genuine"
    ]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Split data into features (X) and labels (y)
X = df['email_text']
y = df['label']

# Split data into training and testing sets (stratify to ensure each class is represented)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# Predict on new email texts
new_emails = [
    "You have won a lottery! Click here to claim your prize!",
    "Urgent: Your account has been compromised, please change your password immediately.",
    "Hi, let's catch up this weekend over coffee."
]

new_emails_vec = vectorizer.transform(new_emails)
new_predictions = model.predict(new_emails_vec)

print("\nNew Email Predictions:")
for email, label in zip(new_emails, new_predictions):
    print(f"Email: {email}\nPredicted Label: {label}\n")
