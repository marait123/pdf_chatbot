from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSDatabase:
    _instance = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FAISSDatabase, cls).__new__(cls)
            cls._instance._db = None
        return cls._instance

    @property
    def db(self) -> Optional[FAISS]:
        """Get the FAISS database instance"""
        return self._db

    def add_documents(self, documents: List) -> None:
        """Add documents to FAISS database"""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            if self._db is None:
                self._db = FAISS.from_documents(documents, embeddings)
                logger.info("Created new FAISS database with documents")
            else:
                self._db.add_documents(documents)
                logger.info("Added documents to existing FAISS database")

        except Exception as e:
            logger.error(f"Error adding documents to FAISS database: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List:
        """Search for similar documents"""
        if self._db is None:
            raise ValueError("Database not initialized. Add documents first.")
        return self._db.similarity_search(query, k=k)

# Usage example:
# db = FAISSDatabase()
# db.add_documents(documents)
# results = db.similarity_search("query")
