
# read and manipulate the dataset
import pandas as pd
# clean text with regular expressions
import re
# natural language processing library
import nltk
from nltk.corpus import stopwords
# saves the trained model into a file to use it later
import pickle
# main machine learning library
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


nltk.download('stopwords')

# load stopwords ONCE as a set 
STOP_WORDS = set(stopwords.words('english'))


# LOAD DATASET
print("Loading dataset...")
df = pd.read_csv('data/IMDB Dataset.csv')
print(df.head())
print(f"Total reviews: {len(df)}")

# CLEAN TEXT
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    useful_words = [w for w in words if w not in STOP_WORDS]
    return ' '.join(useful_words)

print("Cleaning text...")
df['clean_review'] = df['review'].apply(clean_text)

# X = what the model reads
# y = positive or negative
X = df['clean_review']
y = df['sentiment']

# split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data: {len(X_train)}")
print(f"Testing data: {len(X_test)}")

# CONVERT TEXT TO NUMBERS (TF-IDF)
print("Converting text to numbers...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# TRAIN THE MODEL
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# EVALUATE THE MODEL
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nFull report:")
print(classification_report(y_test, y_pred))

# SAVE THE MODEL AND VECTORIZER
print("Saving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model saved successfully")
print("Files created: model.pkl, vectorizer.pkl")