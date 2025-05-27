import random
import json
import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('wordnet')

# Inisialisasi
lemmatizer = WordNetLemmatizer()

# Dataset sederhana intents
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hai", "Halo", "Apa kabar?", "Selamat pagi", "Hai bot"],
            "responses": ["Halo! Ada yang bisa saya bantu seputar hidup sehat?", "Hai! Siap membantu kamu hidup lebih sehat!"]
        },
        {
            "tag": "gizi",
            "patterns": ["Apa itu gizi seimbang?", "Gizi yang baik itu seperti apa?", "Ciri makanan bergizi?"],
            "responses": ["Gizi seimbang adalah asupan nutrisi lengkap yang dibutuhkan tubuh, termasuk karbohidrat, protein, lemak sehat, vitamin dan mineral."]
        },
        {
            "tag": "hidup_sehat",
            "patterns": ["Tips hidup sehat dong", "Gimana cara hidup sehat?", "Cara menjaga tubuh tetap sehat?"],
            "responses": ["Minum cukup air, tidur teratur, olahraga minimal 3 kali seminggu, dan konsumsi makanan bergizi!"]
        },
        {
            "tag": "olahraga",
            "patterns": ["Olahraga yang baik apa?", "Berapa kali olahraga dalam seminggu?", "Saya ingin mulai olahraga"],
            "responses": ["Untuk pemula, kamu bisa mulai dengan jalan kaki 30 menit sehari, 3–5 kali seminggu."]
        },
        {
            "tag": "makan",
            "patterns": ["Makanan sehat apa aja?", "Sarapan sehat itu seperti apa?", "Contoh makan siang bergizi?"],
            "responses": ["Contoh makan sehat: nasi merah, dada ayam kukus, sayur rebus, dan buah sebagai pencuci mulut."]
        }
    ]
}

# Data persiapan
all_patterns = []
all_tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern.lower())
        stemmed = [lemmatizer.lemmatize(word) for word in tokens]
        all_patterns.append(" ".join(stemmed))
        all_tags.append(intent['tag'])

# Vektorisasi & klasifikasi
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(all_patterns)
clf = MultinomialNB()
clf.fit(X, all_tags)

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

# Interaksi CLI
if __name__ == "__main__":
    print("Bot Sehat: Halo! Saya bisa bantu soal hidup sehat dan gizi. Ketik 'quit' untuk keluar.")
    while True:
        inp = input("Kamu: ")
        if inp.lower() == "quit":
            print("Bot Sehat: Sampai jumpa dan tetap sehat!")
            break
        response = chatbot_response(inp)
        print("Bot Sehat:", response)
