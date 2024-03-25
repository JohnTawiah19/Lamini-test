import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
from sentence_transformers import SentenceTransformer
import os
import pandas as pd

registry = EmbeddingFunctionRegistry.get_instance()
stransformer = registry.get("sentence-transformers").create()

class StoreSchema(LanceModel):
    vector: Vector(stransformer.ndims()) = stransformer.VectorField() # type: ignore
    text:str = stransformer.SourceField()
    
table_name = 'my_table'
class Store:     
    def __init__(self,checkpoint ) :
        self.checkpoint = checkpoint
        self.db = None
    
    def create(self, uri = "./lancedb"):
        if not os.path.exists(uri):
            self.db = lancedb.connect(uri)
            self.db.create_table('my_table', schema=StoreSchema.to_arrow_schema())
            print("New database created.")
        else:
            self.db = lancedb.connect(uri)
            print("Connected to existing database.")
        
    def load(self, sentences, query, key):
        if not self.db:
            raise ValueError("Database connection not established.")

        table = self.db.open_table('my_table')
        table.add(pd.DataFrame({"text": sentences }))

        result = table.search(query).limit(5).to_list()
        print("Sentences added to the database.")
        return result
        
    # def query(self, query):    
    #     if not self.db:
    #         raise ValueError("Database connection not established.")
    #     table = self.db
    #     return table.(query).limit(5)
      
        
    # def get_embeddings(self, sentences):
    #     model = SentenceTransformer(self.checkpoint)
    #     return model.encode(sentences)