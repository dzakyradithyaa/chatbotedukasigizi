import random #untuk
import json
import nltk
import os
import joblib
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Unduh NLTK data jika belum penah download
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inisialisasi
lemmatizer = WordNetLemmatizer()

# Load dataset dari file
with open("Dataset.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

# Inisialisasi variabel global
vectorizer_file = "vectorizer.pkl"
model_file = "model.pkl"

# Cek apakah model sudah ada
if os.path.exists(model_file) and os.path.exists(vectorizer_file):
    # Load dari file
    vectorizer = joblib.load(vectorizer_file)
    clf = joblib.load(model_file)

#Jika model belum dibuat
else:
    # Melakukan training dengan data dari dataset
    all_patterns = []
    all_tags = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern.lower())
            stemmed = [lemmatizer.lemmatize(word) for word in tokens]
            all_patterns.append(" ".join(stemmed))
            all_tags.append(intent['tag'])

    # Vektorisasi dan training model
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_patterns)
    clf = MultinomialNB()
    clf.fit(X, all_tags)

    # Menyimpan model dan vectorizer ke file
    joblib.dump(vectorizer, vectorizer_file)
    joblib.dump(clf, model_file)

# Fungsi chatbot
def chatbot_response(user_input):
    tokens = nltk.word_tokenize(user_input.lower())
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    input_transformed = vectorizer.transform([" ".join(lemmas)])
    tag = clf.predict(input_transformed)[0]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Maaf, saya belum mengerti. Bisa coba tanyakan dengan cara berbeda?"

# CLI chatbot
if __name__ == "__main__":
    print("Bot Sehat: Halo! Saya bisa bantu soal hidup sehat dan gizi. Ketik 'quit' untuk keluar.")
    
    # Tampilkan daftar tag
    print("\nTopik yang bisa kamu tanyakan:")
    for intent in intents['intents']:
        print(f"- {intent['tag']}")
    print()

    # Loop interaksi
    while True:
        inp = input("Kamu: ")
        if inp.lower() in ["quit","q","qu"]:
            print("Bot Sehat: Sampai jumpa dan tetap sehat!")
            break
        response = chatbot_response(inp)
        print("Bot Sehat:", response)
