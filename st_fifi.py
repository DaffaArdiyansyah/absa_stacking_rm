import streamlit as st
import gdown # type: ignore
import joblib
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Set halaman
st.set_page_config(page_title="Analisis Sentimen Berbasis Aspek", layout="wide")

st.markdown("<h1 style='text-align: center;'>Aplikasi Analisis Sentimen Berbasis Aspek</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: red;'>Pada Ulasan Rumah Makan</h2>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Metode Stacking Ensemble Learning</h1>", unsafe_allow_html=True)

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

# ID File Google Drive untuk Model
FILES = {
    "KNN": "11IMrIo1XNTI4ZNiaZk4hEzRrczRaeBTw",  # Ganti dengan ID Google Drive untuk KNN
    "SVM": "1AQBE7AVGygTu49SzvljFNNdsGnozJjpx",  # Ganti dengan ID Google Drive untuk SVM
    "Naive Bayes": "18iRs_Y8TWXOTUh08uPmPeK0DiLShQ9wd"  # Ganti dengan ID Google Drive untuk Naive Bayes
}

# Fungsi download file dari Google Drive
@st.cache_resource
def download_model(file_id, output):
    if not os.path.exists(output):
        with st.spinner(f"Mengunduh {output} dari Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    return joblib.load(output)

# Load model meta learning
@st.cache_resource
def load_models():
    return {model: download_model(FILES[model], f"{model.lower().replace(' ', '_')}.pkl") for model in FILES}

models = load_models()

# Load TF-IDF Vectorizer langsung dari file lokal
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load kamus normalisasi
kamus_normalisasi = pd.read_csv('colloquial-indonesian-lexicon.csv')
kamus_dict = dict(zip(kamus_normalisasi['slang'], kamus_normalisasi['formal']))

# Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi preprocessing dengan handling negasi
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [kamus_dict.get(kata, kata) for kata in tokens]  # Normalisasi
    
    # Handling negasi
    negasi = {"tidak", "bukan", "jangan", "belum", "tanpa", "kurang"}
    filtered_tokens = []
    prev_negasi = False
    for word in tokens:
        if word in negasi:
            prev_negasi = True
            filtered_tokens.append(word)
        else:
            if prev_negasi:
                filtered_tokens.append("tidak_" + word)
                prev_negasi = False
            else:
                filtered_tokens.append(word)
    
    stop_words = set(stopwords.words('indonesian')) - negasi  # Stopword tanpa negasi
    tokens = [word for word in filtered_tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Pilih model meta learning
selected_models = st.multiselect("Pilih Salah satu / Semua Meta Learning Model :", list(models.keys()), default=list(models.keys()))

# Input teks ulasan
ulasan = st.text_area("Inputkan Ulasan:", height=150)

if st.button("Proses"):
    if ulasan.strip():
        cleaned_ulasan = preprocess_text(ulasan)
        X_input = vectorizer.transform([cleaned_ulasan]).toarray()
        
        labels = [
            "Makanan_positif", "Makanan_negatif",
            "Layanan_positif", "Layanan_negatif",
            "Fasilitas_positif", "Fasilitas_negatif",
            "Harga_positif", "Harga_negatif"
        ]

        for model_name in selected_models:
            model = models[model_name]
            y_pred = model.predict(X_input)
            y_pred = y_pred[0] if y_pred.ndim > 1 else y_pred
            
            st.subheader(f"Hasil Analisis - {model_name}")
            if all(pred == 0 for pred in y_pred):
                st.write("Tidak ada aspek dan sentimen yang ditemukan!")
            else:
                for i in range(0, len(labels), 2):
                    aspek = labels[i].split("_")[0]
                    positif, negatif = y_pred[i], y_pred[i + 1]
                    
                    if positif == 1 and negatif == 0:
                        sentimen, warna = "Positif", "green"
                    elif positif == 0 and negatif == 1:
                        sentimen, warna = "Negatif", "red"
                    else:
                        sentimen = "Tidak Ada Sentimen"
                    
                    if sentimen != "Tidak Ada Sentimen":
                        st.markdown(f"<p style='color: {warna};'><b>Aspek = {aspek.capitalize()} : Sentimen = {sentimen}</b></p>", unsafe_allow_html=True)
    else:
        st.warning("Masukkan teks ulasan terlebih dahulu!")
