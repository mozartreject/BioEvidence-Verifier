from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print(f"Loading reranker: {model_name}")

        self.model = CrossEncoder(
            model_name,
            max_length=512
        )

        print("Cross-Encoder reranker ready.")

    def rerank(
        self,
        query,
        documents,
        top_k=10
    ):

        if not documents:
            return []

        pairs = [
            (query, doc["text"])
            for doc in documents
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )

        ranked = []

        for doc, score in zip(documents, scores):

            item = dict(doc)

            item["reranker_score"] = float(score)

            ranked.append(item)

        ranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return ranked[:top_k]