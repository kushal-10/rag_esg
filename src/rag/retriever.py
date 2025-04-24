import os
import json
import weaviate
from tqdm import tqdm
import logging
from typing import Dict, List
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join("src", "rag", "retreiver.log"),
    level=logging.INFO,
    filemode='w'
)


def split_text_with_overlap(text: str, chunk_size: int = 200, overlap: int = 20) -> List[Document]:
    """
    Split `text` into word-based chunks of exactly `chunk_size` words,
    with an `overlap` number of words shared between consecutive chunks.
    """
    words = text.split()
    step = chunk_size - overlap
    chunks: List[Document] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        chunks.append(Document(page_content=chunk_text))
    return chunks


class Retrieve():
    """
    Extract relevant passages for each sub-target for a particular PDF,
    using a 200-word sliding window with 20-word overlap.
    """

    def __init__(self, embeddings=OpenAIEmbeddings(), weaviate_client=None, targets_path: str = os.path.join("results", "targets.json")):
        self.embeddings = embeddings
        self.weaviate_client = weaviate_client
        with open(targets_path, "r") as f:
            self.targets = json.load(f)


    def get_db(self, file_path: str) -> WeaviateVectorStore:
        """
        Load the text file, split into overlapping 200-word chunks,
        and build a Weaviate vector store.
        """
        loader = TextLoader(file_path)
        documents = loader.load()
        word_chunks: List[Document] = []
        for doc in documents:
            # apply sliding-window split on each loaded document
            word_chunks.extend(split_text_with_overlap(doc.page_content, chunk_size=200, overlap=20))

        db = WeaviateVectorStore.from_documents(
            word_chunks,
            self.embeddings,
            client=self.weaviate_client
        )
        return db
    

    def retrieve_docs(self, db: WeaviateVectorStore = None, similarity_threshold: float = 0.75) -> Dict:
        """
        Return a dict containing extracted passages of each Sub-Target
        for a given annual report.
        """
        logger.info(f"Retrieving Docs with threshold: {similarity_threshold}")
        retrieved_docs = []
        for i in range(1, 18):
            sub_target_defs = self.targets[str(i)]
            for defn in sub_target_defs:

                # Update defn to include AI + SDG
                # Comment the following line to only include SDG results
                ai_sdg_defn = defn + " Are Artificial Intelligence or similar techonologies (keywords - Machine Learning, Data Science, Computer Vision) used towards this?"

                docs = db.similarity_search_with_score(ai_sdg_defn, k=50)
                count = 0
                for doc, score in docs:
                    if score >= similarity_threshold:
                        count += 1
                        retrieved_docs.append({
                            "goal": i,
                            "sub_target": defn,
                            "page_content": doc.page_content
                        })

        logger.info(f"Retrieved {len(retrieved_docs)} Documents")
        return retrieved_docs


if __name__ == "__main__":
    # Initialize Weaviate client and embeddings
    weaviate_client = weaviate.connect_to_local()
    openai_embeddings = OpenAIEmbeddings()
    retriever = Retrieve(
        embeddings=openai_embeddings,
        weaviate_client=weaviate_client,
        targets_path=os.path.join("results", "targets.json")
    )

    TXT_DIR = os.path.join("data", "txts")
    company = "14.BASF_$42.93 B_Industrials"
    for year in tqdm(os.listdir(os.path.join(TXT_DIR, company)), desc=f"Extracting Docs for: {company}"):
        year_dir = os.path.join(TXT_DIR, company, year)
        if os.path.isdir(year_dir):
            result_file = os.path.join(year_dir, "results.txt")
            db = retriever.get_db(result_file)
            retrieved = retriever.retrieve_docs(db=db, similarity_threshold=0.75)

            save_dir = os.path.join("results", "retrieve", company, year)
            os.makedirs(save_dir, exist_ok=True)
            # with open(os.path.join(save_dir, "passages.json"), "w") as f:
            #     json.dump(retrieved, f, indent=4)
            with open(os.path.join(save_dir, "ai_sdg_passages.json"), "w") as f:
                json.dump(retrieved, f, indent=4)

    weaviate_client.close()
