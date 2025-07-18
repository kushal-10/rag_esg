import os
import json
import logging
from tqdm import tqdm
from typing import List, Dict

import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.docstore.document import Document

# --- Logging setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join("src", "rag", "retriever.log"),
    level=logging.INFO,
    filemode='w'
)

def load_docs_from_embedding_filter(json_path: str) -> List[Document]:
    """
    Loads Document objects from a given embedding_filter.json file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    docs = [
        Document(
            page_content=entry["sentence"],
            metadata={"chunk_id": entry["chunk_id"]}
        ) for entry in data if entry["sentence"].strip()
    ]
    return docs

class Retrieve:
    def __init__(self, embeddings, weaviate_client):
        self.embeddings = embeddings
        self.weaviate_client = weaviate_client

    def get_db(self, json_path: str) -> WeaviateVectorStore:
        docs = load_docs_from_embedding_filter(json_path)
        if not docs:
            logger.warning(f"No docs found in {json_path}")
            return None
        db = WeaviateVectorStore.from_documents(
            docs,
            self.embeddings,
            client=self.weaviate_client
        )
        return db

    def retrieve_docs(self, db: WeaviateVectorStore, keywords: List, similarity_threshold: float = 0.75) -> Dict:

        kw_data = {}
        for keyword in keywords:
            retrieved_docs = []
            docs = db.similarity_search_with_score(keyword, k=1000)
            count = 0
            for doc, score in docs:
                if score >= similarity_threshold:
                    count += 1
                    retrieved_docs.append([score, doc.metadata["chunk_id"], doc.page_content])
            kw_data[keyword] = {
                "total_docs": count,
                "retrieved_docs": retrieved_docs
            }

        return kw_data

if __name__ == "__main__":
    # ---- Setup Weaviate and Embeddings ----
    # If running local, adjust as needed.
    weaviate_client = weaviate.connect_to_local()  # Or connect_to_custom for remote server
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    retriever = Retrieve(
        embeddings=openai_embeddings,
        weaviate_client=weaviate_client
    )

    jsons = []
    for dirname, _, filenames in os.walk("data/textsv2"):
        for filename in filenames:
            if filename.endswith("embedding_filter.json"):
                jsons.append(os.path.join(dirname, filename))

    with open(os.path.join("data", "sdgs.json"), "r", encoding="utf-8") as f:
        sdgs = json.load(f)
    sdg_keywords = list(sdgs.values())

    ai_keywords = [
        "Artificial Intelligence",
        "Machine Learning",
        "Reinforcement Learning",
        "Deep Learning",
        "Computer Vision",
        "Natural Language Processing"
    ]

    for json_path in tqdm(jsons):
        save_path = json_path.replace("embedding_filter.json", "rag_filter_ai.json")
        if not os.path.exists(save_path):
            db = retriever.get_db(json_path)
            if not db:
                continue
            ai_kw_data = retriever.retrieve_docs(db, ai_keywords, similarity_threshold=0.5)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(ai_kw_data, f)

            sdg_kw_data = retriever.retrieve_docs(db, sdg_keywords, similarity_threshold=0.5)
            save_path = json_path.replace("embedding_filter.json", "rag_filter_sdg.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(sdg_kw_data, f)

    weaviate_client.close()
