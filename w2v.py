# Introduction to Word2Vec using Gensim
# @author Alexadre del Fabbro [alexandre.delfabbro@gmail.com]
# D'après le code de Manan Suri sur https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673 [Consulté le 26 novembre 2024]
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample corpus (list of tokenized sentences)
corpus = [
    word_tokenize("The quick brown fox jumps over the lazy dog"),
    word_tokenize("I enjoy learning natural language processing with Gensim"),
]

# Training a Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get the vector for the word "fox"
vector = model.wv['natural']

# Find words most similar to "fox"
similar_words = model.wv.most_similar('natural')
print(similar_words)