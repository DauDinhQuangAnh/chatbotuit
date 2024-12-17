from .base_chunker import BaseChunker
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

class SemanticChunker(BaseChunker):
    def __init__(self, threshold=0.3, embedding_type="tfidf", model="keepitreal/vietnamese-sbert"):
        self.threshold = threshold
        self.embedding_type = embedding_type
        self.model = model
        nltk.download("punkt", quiet=True)
    
    def embed_function(self, sentences):
    
        if self.embedding_type == "tfidf":
            vectorizer = TfidfVectorizer().fit_transform(sentences)
            return vectorizer.toarray()

        else:
            raise ValueError("Unsupported embedding type")
    
        
    def split_text(self, text):
        sentences = nltk.sent_tokenize(text)  # Extract sentences
        sentences = [item for item in sentences if item and item.strip()]
        if not len(sentences):
            return []
        vectors = self.embed_function(sentences)
        similarities = cosine_similarity(vectors)
        chunks = [[sentences[0]]]  
        for i in range(1, len(sentences)):
            sim_score = similarities[i-1, i]

            if sim_score >= self.threshold:
                # If the similarity is above the threshold, add to the current chunk
                chunks[-1].append(sentences[i])
            else:
                # Start a new chunk
                chunks.append([sentences[i]])
        # Join the sentences in each chunk to form coherent paragraphs
        return [' '.join(chunk) for chunk in chunks]