import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
import json
import nltk
from request import ModelRequest

nltk.download("punkt")

class Model:
    def __init__(self, context):
        self.context = context
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.umap_model = UMAP(n_neighbors=15, min_dist=0.0, metric="cosine", random_state=69)
        # self.hdbscan_model = HDBSCAN(min_cluster_size=15, metric="euclidean", prediction_data=True)
        self.topic_model = BERTopic(
            umap_model = self.umap_model,
            # hdbscan_model = self.hdbscan_model,
            vectorizer_model = self.vectorizer_model,
        )

    async def inference(self, request: ModelRequest):
        text = request.text
        try:
            # Encode the text using SentenceTransformer
            corpus_embeddings = self.sentence_model.encode(text)
            
            # Fit the topic model
            topics, probabilities = self.topic_model.fit_transform(text, corpus_embeddings)

            # Get topic information and cluster labels
            df_classes = self.topic_model.get_topic_info()
            cluster_labels, _ = self.topic_model.transform(text, corpus_embeddings)

            df_result = pd.DataFrame({
                "document_text": text,
                "predicted_class_label": cluster_labels,
                "probabilities": probabilities,
            }) 

            # Mapping cluster names to topic labels
            cluster_names_map = dict(zip(df_classes["Topic"], df_classes["Name"]))
            df_result["predicted_class_name"] = df_result["predicted_class_label"].map(cluster_names_map)

            csv_string = df_result.to_csv(index=False)
            
        except Exception as e:
            # Log & print the error
            print(f"Error during inference: {e}")
            return None

        return csv_string

