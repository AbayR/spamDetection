import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, \
    roc_auc_score, roc_curve, confusion_matrix
import string
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'spam.csv'
spam_data = pd.read_csv(file_path)

# Predefined list of common English stopwords
common_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                    'should', 'now'}


# Text preprocessing function
def preprocess_text_simple(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in common_stopwords]
    return ' '.join(tokens)


# Apply simplified preprocessing to the Message column
spam_data['Message'] = spam_data['Message'].apply(preprocess_text_simple)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spam_data['Message'], spam_data['Category'], test_size=0.2,
                                                    random_state=42)

# Create a dictionary of pipelines for each model
models = {
    'MultinomialNB': Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB(alpha=0.1))
    ]),
    'SVM': Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', SVC(probability=True))
    ]),
    'RandomForest': Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', RandomForestClassifier())
    ]),
    'KNN': Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', KNeighborsClassifier())
    ])
}

# Train and evaluate each model
results = {}
for model_name, pipeline in models.items():
    print(f'Training {model_name}...')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline['classifier'], 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    conf_matrix = confusion_matrix(y_test, y_pred)

    results[model_name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': classification_report(y_test, y_pred)
    }

    if roc_auc is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label='spam')
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Save the trained model and vectorizer
joblib.dump(models['MultinomialNB'], 'spam_detector_pipeline.pkl')

# Output results and plot confusion matrix
for model_name, metrics in results.items():
    print(f'\n{model_name} Results:')
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"ROC AUC: {metrics['roc_auc']}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(metrics['classification_report'])

    plt.figure(figsize=(10, 7))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 7))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
