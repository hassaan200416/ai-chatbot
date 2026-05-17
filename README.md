# AI Customer Support Chatbot

A full-stack AI-powered customer support chatbot built as a semester project for the **Artificial Intelligence course at Bahria University (Spring 2026)**.

The system classifies user intent from natural language input using two ML models — Naive Bayes and an Artificial Neural Network — and returns context-appropriate responses in real time.

---

## Features

- Dual ML models: Multinomial Naive Bayes and Keras MLP (ANN)
- 27 customer support intent classes from the Bitext dataset
- Real-time intent classification with confidence scores
- Full NLP preprocessing pipeline (NLTK tokenisation, stopword removal, lemmatisation)
- TF-IDF + Word-level feature extraction
- Persistent chat history via Supabase (PostgreSQL)
- Clean chat UI with model switcher, session stats, and intent badges
- 38 automated tests (unit + integration)

---

## Tech Stack

| Layer     | Technology                                     |
| --------- | ---------------------------------------------- |
| Language  | Python 3.11                                    |
| NLP       | NLTK                                           |
| Features  | Scikit-learn TF-IDF                            |
| ML Models | Scikit-learn Naive Bayes, Keras/TensorFlow ANN |
| Backend   | Flask, Flask-CORS                              |
| Database  | Supabase (PostgreSQL)                          |
| Frontend  | HTML5, CSS3, Vanilla JavaScript                |

---

## Project Structure

ai-chatbot/
│
├── backend/
│ ├── app.py # Flask app + all routes
│ ├── config.py # Environment config
│ ├── preprocessor.py # NLP cleaning pipeline
│ ├── predictor.py # Model inference
│ ├── response_map.py # Intent → response text
│ ├── history.py # Supabase save/read
│ └── saved_models/ # Trained model files (gitignored)
│
├── ml/
│ ├── data/ # Dataset CSV (gitignored)
│ ├── train_nb.py # Train Naive Bayes
│ ├── train_ann.py # Train ANN
│ └── evaluate.py # Full evaluation report
│
├── frontend/
│ ├── index.html # Chat UI
│ ├── style.css # Styling
│ └── chat.js # Fetch API calls
│
├── tests/
│ ├── test_app.py # Route integration tests
│ └── test_preprocessor.py # Preprocessing unit tests
│
├── .env.example # Environment variable template
├── requirements.txt
└── README.md

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-chatbot.git
cd ai-chatbot
```

### 2. Create virtual environment with Python 3.11

```bash
py -3.11 -m venv venv
venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate          # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your Supabase URL and key:
FLASK_ENV=development
FLASK_SECRET_KEY=your-secret-key-here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
MODEL_TYPE=ann

### 6. Create the Supabase table

Run this SQL in your Supabase SQL Editor:

```sql
CREATE TABLE chat_history (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id TEXT NOT NULL,
  user_message TEXT NOT NULL,
  bot_response TEXT NOT NULL,
  predicted_intent TEXT,
  confidence FLOAT,
  model_used TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_session_id ON chat_history(session_id);
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow backend access" ON chat_history FOR ALL USING (true);
```

### 7. Download the dataset

Download the Bitext Customer Support Intent Dataset from:
https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants

Place the CSV file at `ml/data/bitext_dataset.csv`.

### 8. Train the models

```bash
python ml/train_nb.py
python ml/train_ann.py
```

### 9. Start the backend

```bash
cd backend
python app.py
```

### 10. Open the frontend

Open `frontend/index.html` in your browser.

---

## Model Performance (Test Set — 3,231 samples)

| Metric    | Naive Bayes | ANN (MLP) |
| --------- | ----------- | --------- |
| Accuracy  | 98.58%      | 98.45%    |
| Precision | 98.59%      | 98.47%    |
| Recall    | 98.58%      | 98.45%    |
| F1-Score  | 98.56%      | 98.44%    |

---

## API Endpoints

### `POST /api/chat`

```json
Request:  { "message": "I want to cancel my order", "session_id": "uuid" }
Response: { "response": "...", "intent": "cancel_order", "confidence": 0.99, "model_used": "ann" }
```

### `GET /api/history?session_id=<uuid>`

```json
Response: { "session_id": "uuid", "history": [ { "user_message": "...", "bot_response": "..." } ] }
```

### `GET /api/health`

```json
Response: { "status": "ok", "model": "ann", "env": "development" }
```

---

## Running Tests

```bash
pytest tests/ -v
```

38 tests — 18 unit tests (preprocessor) + 20 integration tests (API routes).

---

## Dataset

**Bitext Customer Support Intent Dataset**

- 21,534 utterances across 27 intent classes
- Split: 70% train / 15% validation / 15% test
- Source: https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants

---

## Contributors

- Group Project — Bahria University, Spring 2026
- Course: Artificial Intelligence
