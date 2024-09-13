# pip install transformers sentencepiece torch sentence-transformers
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SentenceSimilarity:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SentenceSimilarity, cls).__new__(cls)
        return cls._instance

    def __init__(self, dataset_path: str = 'dataset.csv', transformer_model_name: str = 'cointegrated/rubert-tiny2', embedding_file: str = 'embeddings.npy',
                 threshold_upper: float = 0.95,threshold_lower: float = 0.95):
        if not hasattr(self, '_initialized'):
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
            self.model = AutoModel.from_pretrained(transformer_model_name)
            # Uncomment this line if you have a GPU
            # self.model.cuda()
            self.sentence_transformer = SentenceTransformer(transformer_model_name)
            df_combined = pd.read_csv(dataset_path)
            self.sentences = df_combined['source'].tolist()
            self.embeddings = self._load_or_calculate_embeddings(embedding_file)
            self.threshold_upper = threshold_upper
            self.threshold_lower = threshold_lower
            self._initialized = True

    def _load_or_calculate_embeddings(self, embedding_file: str):
        try:
            embeddings = np.load(embedding_file)
        except FileNotFoundError:
            embeddings = self.sentence_transformer.encode(self.sentences)
            np.save(embedding_file, embeddings)
        return embeddings

    def _embed_sentence_cls(self, text: str):
        t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()

    def _filter_similar_sentences(self, top_sentences, query_sentence):
        unique_sentences = []
        seen = set()
        for sent, sim in top_sentences:
            if sent not in seen and sent != query_sentence and sim < self.threshold_upper and sim > self.threshold_lower:
                unique_sentences.append((sent, sim))
                seen.add(sent)
        return unique_sentences

    def find_similar(self, sentence: str, top_n: int = 3):
        query_embedding = self._embed_sentence_cls(sentence)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-10:][::-1]  # Get top 10 to filter later
        top_sentences = [(self.sentences[i], similarities[i]) for i in top_indices]
        filtered_sentences = self._filter_similar_sentences(top_sentences, sentence)
        return filtered_sentences[:top_n]


similarity_model = SentenceSimilarity(
        dataset_path='ml/sensim/dataset.csv',
        transformer_model_name='cointegrated/rubert-tiny2',
        embedding_file='ml/sensim/embeddings.npy',
        threshold_upper=0.95,
        threshold_lower=0.65,
    )

# if __name__ == "__main__":
#     similarity_model = SentenceSimilarity(
#         dataset_path='ml\sensim\dataset.csv',
#         transformer_model_name='cointegrated/rubert-tiny2',
#         embedding_file='ml\sensim\embeddings.npy',
#         threshold_upper=0.95,
#         threshold_lower=0.65,
#     )

#     sentence = "Сейчас воблу совсем не ловят:"
#     similar_sentences = similarity_model.find_similar(sentence)

#     # for s, sim in similar_sentences:
#     #     print(f"Sentence: {s}, Similarity: {sim:.4f}")
