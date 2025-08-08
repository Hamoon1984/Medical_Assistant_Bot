import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np
import math


# =======================
# just a fancy print :)
# =======================
def fancy_print(text: str, color="blue", size=60):
   ansi_color = "\033[94m"
   if color == "green":
       ansi_color = "\033[95m"
   elif color == "blue":
       ansi_color = "\033[94m"
   else:
       raise Exception(f"Color {color} not supported")

   end_color = "\033[0m"
   str_len = len(text)
   padding = math.ceil((size - str_len) / 2)
   header_len = padding * 2 + str_len + 2
   border = "#" * header_len
   message = "#" * padding + " " + text + " " + "#" * padding
   print(f"{ansi_color}\n{border}\n{message}\n{border}\n{end_color}")

# =======================
# Load Dataset
# =======================
fancy_print("Load Dataset!")
print("Data file and location: data/mle_screening_dataset.csv")
# It is possible to add more data and/or enrich the data, but I did not have much time today to research and find open-source data to do that.
data = pd.read_csv("data/mle_screening_dataset.csv")


fancy_print("Preprocess data!")
# Removing leading and trailing whitespace and normalizing internal spaces
print("Removing leading and trailing whitespace and normalizing internal spaces")
data['answer'] = data['answer'].str.strip().str.replace(r'\s+', ' ', regex=True)
# Remove duplicates to prevent finding the exact answer and leaking data between train and test dataset
print("Removed duplicates")
data.drop_duplicates(subset=["question", "answer"], inplace=True)
# Remove NAN answers
print("Removed NAN answers")
data.dropna(subset=['answer'], inplace=True)
#  Reset index
data.reset_index(drop=True, inplace=True)

# =======================
# Split Data
# =======================
fancy_print("Training data info!")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print("Train data has " + str(train_data.shape[0]) + " observations")
print("Test data has " + str(test_data.shape[0]) + " observations")
# =======================
# TF-IDF Vectorization
# =======================
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(train_data["question"])
eval_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_eval_matrix = eval_vectorizer.fit_transform(data["question"])


# =======================
# Function to Get Best Answer
# =======================
def get_answer(user_question):
    """
    Finds the most similar question in the training data and returns the corresponding answer.
    """
    user_vec = vectorizer.transform([user_question])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    # Here, we used argmax to find the best answer. Another method (If we wanted to use a language model) exists.
    # We could get the top N best answers and pass them all through a small LLM as augmented (enriched) prompt for sumerization to get a better answer.
    best_idx = similarity_scores.argmax()
    pred_answer = train_data.iloc[best_idx]["answer"]
    return pred_answer, similarity_scores[best_idx]

# =======================
# Model Evaluation
# =======================
def evaluate_model():
    ranks = []
    correct = 0
    for _, row in test_data.iterrows():
        true_q = row["question"]
        true_a = row["answer"]

        pred_a, _ = get_answer(true_q)

        # Rank evaluation (MRR)
        user_vec = eval_vectorizer.transform([true_q])
        #similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
        similarity_scores = cosine_similarity(user_vec, tfidf_eval_matrix).flatten()
        sorted_indices = np.argsort(similarity_scores)[::-1]
        #rank = np.where(train_data.iloc[sorted_indices]["answer"].values == true_a)[0]
        rank = np.where(data.iloc[sorted_indices]["answer"].values == true_a)[0]
        if len(rank) > 0:
            ranks.append(1 / (rank[0] + 1))

    mrr = np.mean(ranks) if ranks else 0
    return mrr

# Evaluate and print results
mrr = evaluate_model()

fancy_print("Mean Reciprocal Rank for evaluation")
print(f"Mean Reciprocal Rank (MRR): {mrr:.2f}")

# =======================
# Example Interactions
# =======================
examples = [
    "What is Glaucoma?",
    "How to prevent High Blood Pressure?",
    "What are the symptoms of High Blood Pressure?"
]

fancy_print("Example Interactions:")

for q in examples:
    ans, score = get_answer(q)
    print(f"\nUser: {q}\nBot: {ans}")
    print("************************")
