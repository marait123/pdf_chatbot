from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Optional
import logging
import os
from config import get_config

config = get_config()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSDatabase:
    _instance = None
    _db = None
    _embeddings = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FAISSDatabase, cls).__new__(cls)
            cls._instance._db = None
            cls._instance._embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            cls._instance._load_db()
        return cls._instance

    def _load_db(self) -> None:
        """Load FAISS database from disk if it exists"""
        try:
            if os.path.exists(config.FAISS_DB_PATH):
                self._db = FAISS.load_local(
                    config.FAISS_DB_PATH,
                    self._embeddings,
                    allow_dangerous_deserialization=True  # Add this parameter
                )
                logger.info(f"Loaded existing FAISS database from {config.FAISS_DB_PATH}")
            else:
                logger.info(f"FAISS database not found at {config.Faiss_DB_PATH}")

        except Exception as e:
            logger.error(f"Error loading FAISS database: {str(e)}")
            self._db = None

    def _save_db(self) -> None:
        """Save FAISS database to disk"""
        try:
            if self._db is not None:
                # Ensure directory exists
                os.makedirs(os.path.dirname(config.FAISS_DB_PATH), exist_ok=True)
                self._db.save_local(config.FAISS_DB_PATH)
                logger.info(f"Saved FAISS database to {config.FAISS_DB_PATH}")
        except Exception as e:
            logger.error(f"Error saving FAISS database: {str(e)}")

    @property
    def db(self) -> Optional[FAISS]:
        """Get the FAISS database instance"""
        return self._db

    def add_documents(self, documents: List) -> None:
        """Add documents to FAISS database and persist to disk"""
        try:
            if self._db is None:
                self._db = FAISS.from_documents(documents, self._embeddings)
                logger.info("Created new FAISS database with documents")
            else:
                self._db.add_documents(documents)
                logger.info("Added documents to existing FAISS database")

            # Save after adding documents
            self._save_db()

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
