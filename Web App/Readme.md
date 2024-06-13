# Dialect Prediction Flask App

This Flask application predicts the dialect of Arabic text input using a pre-trained Multinomial Naive Bayes model. It features text preprocessing, model prediction, and a web interface for user interaction.

## Key Features

1. **Text Preprocessing**:
   - Cleans input text by removing numbers and non-Arabic characters.
   - Removes Arabic stop words to focus on relevant words for dialect prediction.

2. **Model Prediction**:
   - Uses a TF-IDF vectorizer to transform the preprocessed text.
   - Predicts the dialect using a pre-trained Multinomial Naive Bayes model.
   - Maps predicted labels to full dialect names.

3. **Web Interface**:
   - HTML interface for users to input text and view the predicted dialect.

## How It Works

1. **User Input**:
   - Users enter Arabic text on the web page and submit it.

2. **Backend Processing**:
   - The text is sent to the Flask backend via a POST request.
   - Text is preprocessed and passed through the model for prediction.
   - Predicted label is mapped to the full dialect name.

3. **Result Display**:
   - The web page displays the original text and the predicted dialect.

## Technologies Used

- **Flask**: Web framework for building the application.
- **scikit-learn**: For the machine learning model and TF-IDF vectorizer.
- **NLTK**: For text preprocessing.
- **HTML/CSS**: For the web interface.

## Project Files

- **app.py**: Main Flask application file with routes and logic.
- **templates/index.html**: HTML template for the web interface.
- **finalnb_path.pkl**: Pre-trained Multinomial Naive Bayes model.
- **vectorizer_path.pkl**: TF-IDF vectorizer.
- **label_encoder_path.pkl**: Label encoder for mapping labels to dialects.

dialect-prediction-app/
│
├── templates/
│   └── index.html
├── finalnb_path.pkl
├── vectorizer_path.pkl
├── label_encoder_path.pkl
├── app.py
└── requirements.txt

## Installation and Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/dialect-prediction-app.git
   cd dialect-prediction-app
