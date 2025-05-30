# 🚀 Text Classification with Apache Spark MLlib

This project demonstrates how to preprocess text data using Apache Spark's MLlib pipeline. It converts raw comments into sparse TF-IDF vectors for use in machine learning models such as logistic regression or random forest.

## 📁 Files

- `train.csv`: Input dataset containing text and toxicity label columns.
- `spark_text_classification.py`: Main Spark pipeline script.
- The dataset can be found at https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

## 🧰 Technologies Used

- Apache Spark (PySpark)
- Spark MLlib
- Tokenizer, HashingTF, IDF
- CSV data loading

## 📊 Dataset Format

The input `train.csv` file must contain at least:
- `comment_text`: The raw text of a user comment.
- Toxicity columns such as `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

## 🛠️ How It Works

1. **Load CSV** and drop missing values from `comment_text`.
2. **Create a binary label** indicating whether a comment is toxic (if any of the toxicity flags are 1).
3. **Tokenize** the `comment_text` into words.
4. **Convert words to sparse vectors** using `HashingTF`.
5. **Apply IDF weighting** to compute final TF-IDF feature vectors.
6. **Filter null labels** before model training or evaluation.

## 🔍 Example Output

The final output is a DataFrame with columns:
- `comment_text`: Original text.
- `label`: Binary toxicity label (0 or 1).
- `features`: A 10,000-dimensional sparse TF-IDF vector.

## ▶️ Sample Usage

```bash
./test.sh
