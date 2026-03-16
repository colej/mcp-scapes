from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self) -> None:
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
