import random
from faker import Faker
from sentence_transformers import SentenceTransformer, util
import time

# This file contain the ML model to do string semantic comparision between SumDB with user question

# Initialize Faker for generating random sentences
fake = Faker()

NUM_SENTENCES = 10000
MODEL_NAME = 'all-MiniLM-L6-v2' # Refer to https://www.sbert.net/docs/pretrained_models.html
print(f'Generating {NUM_SENTENCES} data')
sentences = [fake.sentence() for _ in range(NUM_SENTENCES)] + ['a sentence against', 'a sentence to compare']

# Choose a pre-trained SentenceTransformer model
model = SentenceTransformer(MODEL_NAME)

# Single sentence to compare against
single_sentence = "This is a sentence to compare against."

# Efficiently encode all sentences into embeddings
print('Start generate embeddings')
start = time.perf_counter()
embeddings = model.encode(sentences, convert_to_tensor=True)
single_embedding = model.encode(single_sentence, convert_to_tensor=True)
print(f'Time taken: {time.perf_counter() - start}')

# Calculate cosine similarity using PyTorch
print('Finding cos similarity')
similarities = util.pytorch_cos_sim(single_embedding, embeddings)
print(f'Result: {similarities}')  # Output: Similarities between single_sentence and each sentence in 'sentences' 
# a tensor of size (1, NUM_SENTENCES)
