### Spam Detection System

This project aims to build a spam detection system using various machine learning algorithms. The system preprocesses email messages, extracts features using TF-IDF vectorization, and trains multiple classifiers to identify spam messages. The best performing model is selected based on evaluation metrics and used for prediction.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)
- [Endpoints](#endpoints)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/spam-detection-system.git
    cd spam-detection-system
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the FastAPI application:**
    ```sh
    uvicorn main:app --reload
    ```

2. **Access the application:**
    Open your web browser and go to `http://127.0.0.1:8000`.

3. **Use the endpoints:**
    - Register a new user
    - Login to get an access token
    - Use the access token to make authenticated requests

## Model Training

1. **Load and preprocess the dataset:**
    The dataset `spam.csv` is loaded and preprocessed to convert text data to lowercase, remove punctuation, and tokenize the text while removing stopwords.

2. **Train multiple models:**
    The script trains four different models: Multinomial Naive Bayes, SVM, Random Forest, and KNN. The models are trained using a TF-IDF vectorizer to extract features from the email messages.

3. **Save the best model:**
    The best performing model is saved to a file `spam_detector_pipeline.pkl` using `joblib`.

## Model Evaluation

The models are evaluated based on several metrics:
- Accuracy
- F1 Score
- Precision
- Recall
- ROC AUC
- Confusion Matrix

## Visualization

The script generates visualizations for the following:
- ROC curves for each model
- Confusion matrix for each model
- Heatmap of TF-IDF feature scores for spam and ham messages

## Endpoints

### `/register`

**Description:** Register a new user.

**Method:** POST

**Request Body:**
```json
{
    "email": "user@example.com",
    "password": "password"
}
```

**Response:**
```json
{
    "msg": "User created successfully"
}
```

### `/token`

**Description:** Login to get an access token.

**Method:** POST

**Request Body:**
```json
{
    "username": "user@example.com",
    "password": "password"
}
```

**Response:**
```json
{
    "access_token": "your-access-token",
    "token_type": "bearer"
}
```

### `/predict_public`

**Description:** Predict spam without authentication.

**Method:** POST

**Request Body:**
```json
{
    "text": "your-email-message"
}
```

**Response:**
```json
{
    "prediction": "spam/ham"
}
```

### `/predict`

**Description:** Predict spam with authentication.

**Method:** POST

**Request Headers:**
```http
Authorization: Bearer your-access-token
```

**Request Body:**
```json
{
    "text": "your-email-message"
}
```

**Response:**
```json
{
    "prediction": "spam/ham"
}
```

### `/custom_rules`

**Description:** Add a custom rule for a user.

**Method:** POST

**Request Headers:**
```http
Authorization: Bearer your-access-token
```

**Request Body:**
```json
{
    "rule": "your-custom-rule"
}
```

**Response:**
```json
{
    "status": "rule added"
}
```

### `/report`

**Description:** Get the prediction report for a user.

**Method:** GET

**Request Headers:**
```http
Authorization: Bearer your-access-token
```

**Response:** HTML page with the report.
