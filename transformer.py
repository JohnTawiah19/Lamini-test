from cachetools import cached
from lancedb.embeddings import register,TextEmbeddingFunction
from lancedb.util import attempt_import_or_raise
import sentence_transformers

@register("sentence-transformers")
class SentenceTransformerEmbeddings(TextEmbeddingFunction):
    name: str = "all-MiniLM-L6-v2"
    # set more default instance vars like device, etc.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ndims = None

    def generate_embeddings(self, texts):
        return self._embedding_model().encode(list(texts), ...).tolist()

    def ndims(self):
        if self._ndims is None:
            self._ndims = len(self.generate_embeddings("foo")[0])
        return self._ndims

    @cached(cache={}) 
    def _embedding_model(self):
        return sentence_transformers.SentenceTransformer(self.name)