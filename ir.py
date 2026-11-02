# ============================================================
# PRACTICAL 1: TEXT PREPROCESSING
# ============================================================

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

text = """Information retrieval systems retrieve useful information
from large collections of documents. Retrieval systems process
queries and documents efficiently."""

import nltk

nltk.download('punkt_tab')

tokens = word_tokenize(text.lower())

print("Original tokens:")
print(tokens)

stop_words = set(stopwords.words("english"))

filtered = [
    w for w in tokens
    if w.isalpha() and w not in stop_words
]

print("After stop-word removal:")
print(filtered)

stemmer = PorterStemmer()

stems = [stemmer.stem(w) for w in filtered]

print("After stemming:")
print(stems)

print("Vocabulary sizes")

print("Before filtering:",
      len(set(w for w in tokens if w.isalpha())))

print("After filtering:",
      len(set(filtered)))

print("After stemming:",
      len(set(stems)))

print("Original tokens:")
print(tokens)

print("\nAfter stop-word removal:")
print(filtered)

print("\nAfter stemming:")
print(stems)

print("\nVocabulary sizes:")
print("Before filtering:",
      len(set(w for w in tokens if w.isalpha())))
print("After filtering:",
      len(set(filtered)))
print("After stemming:",
      len(set(stems)))


# ============================================================
# PRACTICAL 2: INVERTED INDEX
# ============================================================

import re
from collections import defaultdict

documents = {
    "D1": "The quick brown fox jumped over the lazy dog.",
    "D2": "The lazy dog slept in the sun.",
    "D3": "Information retrieval uses an inverted index."
}

def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())

index = defaultdict(set)

for doc_id, text in documents.items():
    terms = set(tokenize(text))
    for term in terms:
        index[term].add(doc_id)

print("INVERTED INDEX")

for term in sorted(index):
    print(f"{term:15} -> {sorted(index[term])}")

query = input("\nEnter a term: ").strip().lower()

print("Retrieved documents:", sorted(index.get(query, set())))

q1 = input("\nEnter first term for AND query: ").strip().lower()
q2 = input("Enter second term for AND query: ").strip().lower()

result = index.get(q1, set()) & index.get(q2, set())

print("AND result:", sorted(result))


# ============================================================
# PRACTICAL 3: BAYESIAN NETWORK
# ============================================================

%pip install pgmpy

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

df = pd.read_csv("heart_disease.csv")

df = df[["age", "sex", "cp", "chol", "target"]].dropna().copy()

df["age"] = pd.cut(
    df["age"], bins=[0, 40, 55, 100],
    labels=["young", "middle", "senior"]
)

df["chol"] = pd.cut(
    df["chol"], bins=[0, 200, 240, 1000],
    labels=["normal", "borderline", "high"]
)

for col in ["sex", "cp", "target"]:
    df[col] = df[col].astype(str)

train, test = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["target"]
)

model = DiscreteBayesianNetwork([
    ("target", "cp"),
    ("target", "chol"),
    ("target", "age"),
    ("sex", "cp")
])

estimator = MaximumLikelihoodEstimator(model, train)
model.add_cpds(*estimator.get_parameters())

print(model.check_model())
inference = VariableElimination(model)

inference = VariableElimination(model)

predictions = []

for _, row in test.iterrows():
    evidence = {
        "age": row["age"],
        "sex": row["sex"],
        "cp": row["cp"],
        "chol": row["chol"]
    }

    q = inference.query(
        variables=["target"],
        evidence=evidence,
        show_progress=False
    )

    predicted = str(q.state_names["target"][q.values.argmax()])
    predictions.append(predicted)

print("Accuracy:", accuracy_score(test["target"], predictions))
print(classification_report(test["target"], predictions))


# ============================================================
# PRACTICAL 4: SPAM CLASSIFICATION USING NAIVE BAYES
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import pandas as pd

df = pd.read_csv("spam.csv", encoding="latin-1")

print("Column names:", df.columns.tolist())
print(df.head())

import pandas as pd

df = pd.read_csv("spam.csv", encoding="latin-1")

df = df.rename(columns={
    "v1": "label",
    "v2": "message"
})

df = df[["label", "message"]].dropna()

df.to_csv("spam_dataset.csv", index=False, encoding="utf-8")

print(df.head())
print(df.columns.tolist())

df = pd.read_csv("spam_dataset.csv").dropna()

df["label"] = df["label"].str.lower().str.strip()

print(df.shape)
print(df["label"].value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

print("Model trained successfully!")

pred = model.predict(X_test)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("Accuracy:", accuracy_score(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

samples = [
    "Congratulations! You have won a free prize. Claim now.",
    "Please attend the project review meeting at 11 AM."
]

print("\nSample predictions:")

for msg, label in zip(samples, model.predict(samples)):
    print(label, "->", msg)


# ============================================================
# PRACTICAL 5: PAGERANK ALGORITHM
# ============================================================

import numpy as np

graph = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C"]
}

pages = list(graph)
N = len(pages)
d = 0.85

rank = {p: 1 / N for p in pages}

for iteration in range(100):

    new_rank = {p: (1 - d) / N for p in pages}

    for src in pages:
        if graph[src]:
            share = d * rank[src] / len(graph[src])

            for dest in graph[src]:
                new_rank[dest] += share

        else:
            # Dangling node
            share = d * rank[src] / N

            for dest in pages:
                new_rank[dest] += share

    difference = sum(
        abs(new_rank[p] - rank[p]) for p in pages
    )

    rank = new_rank

    if difference < 1e-10:
        break

print("Iterations:", iteration + 1)

print("PageRank scores:")

for page, score in sorted(
    rank.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(page, ":", round(score, 6))

print("Sum of scores:", round(sum(rank.values()), 6))
