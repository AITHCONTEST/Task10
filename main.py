# pip install transformers sentencepiece torch sentence-transformers
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]  #cls
    embeddings = torch.nn.functional.normalize(embeddings)
    # embeddings = model_output.last_hidden_state #avg
    # embeddings = torch.mean(embeddings, dim=1)  # Average embeddings of all tokens
    # embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

def calculate_default_embeddings(model2, sentences):
    try:
        embeddings = np.load('embeddings.npy')
    except FileNotFoundError:
        embeddings = model2.encode(sentences)
        np.save('embeddings.npy', embeddings)
        print("Embeddings saved to disk.")
    return embeddings

def filter_similar_sentences(top_sentences, query_sentence, threshold=0.95):
    unique_sentences = []
    seen = set()
    for sent, sim in top_sentences:
        if sent not in seen and sent != query_sentence and sim < threshold:
            unique_sentences.append((sent, sim))
            seen.add(sent)
    return unique_sentences

def find_similar_sentences(sentence, model1, tokenizer, all_embeddings, sentences):
    query_embedding = embed_bert_cls(sentence, model1, tokenizer)
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    top10_indices = np.argsort(similarities)[-10:][::-1]
    top10_sentences = [(sentences[i], similarities[i]) for i in top10_indices]
    filtered_sentences = filter_similar_sentences(top10_sentences, sentence, threshold=0.95)
    return filtered_sentences[:3]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model1 = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    # model1.cuda()  # Uncomment if you have a GPU
    model2 = SentenceTransformer('cointegrated/rubert-tiny2')

    df_combined = pd.read_csv('dataset.csv')
    sentences = df_combined['source'].tolist()
    all_embeddings = calculate_default_embeddings(model2, sentences)

    sentence = "Сейчас воблу совсем не ловят:"
    similar_sentences = find_similar_sentences(sentence, model1, tokenizer, all_embeddings, sentences)
    for s, sim in similar_sentences:
        print(f"Sentence: {s}, Similarity: {sim:.4f}")
