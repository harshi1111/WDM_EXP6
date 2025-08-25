### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

```
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# The NLTK downloader will try to download the necessary packages.
# This might require an internet connection the first time it's run.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Sample documents stored in a dictionary
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

# Preprocessing function to tokenize and remove stopwords/punctuation
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return " ".join(tokens)

# Preprocess documents and store them in a dictionary
preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

# Construct TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

# Calculate cosine similarity between query and documents
def search(query, tfidf_matrix, tfidf_vectorizer):
    # Preprocess the query
    query_processed = preprocess_text(query)
    
    # Transform the query into a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query_processed])
    
    # Calculate cosine similarity between the query vector and all document vectors
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Pair documents with their similarity scores
    doc_ids = list(documents.keys())
    search_results = []
    for i, score in enumerate(cosine_similarities):
        # Only include results with a similarity score > 0
        if score > 0:
            doc_id = doc_ids[i]
            original_doc = documents[doc_id]
            search_results.append((doc_id, original_doc, score))
            
    # Sort results by similarity score in descending order
    search_results.sort(key=lambda x: x[2], reverse=True)
    
    return search_results

# Get input from user
query = input("Enter your query: ")

# Perform search
search_results = search(query, tfidf_matrix, tfidf_vectorizer)

# Display search results
print("\nQuery:", query)
if search_results:
    for i, result in enumerate(search_results, start=1):
        print(f"\nRank: {i}")
        print("Document ID:", result[0])
        print("Document:", result[1])
        print(f"Similarity Score: {result[2]:.4f}")
        print("----------------------")

    # Get the highest rank cosine score
    highest_rank_score = max(result[2] for result in search_results)
    print("The highest rank cosine score is:", f"{highest_rank_score:.4f}")
else:
    print("No relevant documents found for your query.")
```

### Output:

<img width="542" height="182" alt="image" src="https://github.com/user-attachments/assets/d4a2f050-dfae-453f-ace1-0222c7e8dbb5" />

### Result:

Hence, Successfully implemented Information Retrieval Using Vector Space Model in Python.


