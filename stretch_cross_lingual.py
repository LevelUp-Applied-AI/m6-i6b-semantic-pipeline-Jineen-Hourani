import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import arabic_reshaper
from bidi.algorithm import get_display

def get_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(emb)
    return np.array(embeddings)

def fix_text(text):
    reshaped_text = arabic_reshaper.reshape(text) # ربط الحروف
    return get_display(reshaped_text)

# 1. Load Data
df = pd.read_csv("data/climate_articles.csv")

# 2. Select 10 English and 10 Arabic texts (Ideally same topics)
en_texts = df[df['language'] == 'en']['text'].head(10).tolist()
ar_texts = df[df['language'] == 'ar']['text'].head(10).tolist()
all_texts = en_texts + ar_texts

# 3. Load Multilingual Model
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 4. Compute Embeddings
print("Computing embeddings for 20 texts...")
embeddings = get_embeddings(all_texts, tokenizer, model)

# 5. Compute Similarity Matrix
sim_matrix = cosine_similarity(embeddings)

# 6. Visualization
labels = [fix_text(t[:30] + "...") for t in all_texts]
plt.figure(figsize=(15, 12))
sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels, annot=False, cmap='YlGnBu')
plt.xticks(rotation=45, ha='right')
plt.title(fix_text("Cross-Lingual Similarity: English vs Arabic"))
plt.tight_layout()
plt.savefig("cross_lingual_heatmap_fixed.png")