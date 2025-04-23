import os
import json
import weaviate
from tqdm import tqdm
import logging
from typing import Dict
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join("src", "rag", "retreiver.log"),
    level=logging.INFO
    )

class Retrieve():

    """
    Extract Relevant passages for each sub-target for a particular PDF
    """

    def __init__(self, embeddings = OpenAIEmbeddings(), weaviate_client = None, targets_path: str = os.path.join("results", "targets.json")):
        """
        Args:
            embeddings: Embedding model to generate Weaviate docs, defaults to OpenAI Embeddings
            weaviate_client: Local docker image for Weaviate
            targets_path: Path to a JSON file containing definitions of sub-targets
        """
        self.embeddings = embeddings
        self.weaviate_client = weaviate_client
        with open(targets_path, 'r') as f:
            self.targets = json.load(f)


    def get_db(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> WeaviateVectorStore:
        """
        Return a Weaviate Database by splitting the given text file
        Args:
            file_path: The path of the text file of a given annual report
            chunk_size: The size of chunks in characters to form a document
            chunk_overlap: Overlap of characters between consecutive documents

        Returns:
            db: Weaviate Database based on sepearate documents from the text file
        """

        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        db = WeaviateVectorStore.from_documents(docs, self.embeddings, client=self.weaviate_client)

        return db
    
    
    def retrieve_docs(self, db: WeaviateVectorStore = None, similarity_threshold: float = 0.75) -> Dict:
        """
        Return a Dict object containing extracted passages of each Sub-Target for a given annual report

        Args:
            db: Weaviate Database based on sepearate documents from a given text file 
        
        Returns:
            retrieved_docs: A dict containing relevant passages of each Sub-Target
        """

        logger.info(f"Retrieving Docs with threshold : {similarity_threshold}")
        retrieved_docs = []
        for i in range(1, 18):
            # Iterate over each Goal
            sub_target_def = self.targets[str(i)]
            for defn in sub_target_def:
                retrieved_count = 0
                docs = db.similarity_search_with_score(defn, k=50) # Retrieve top 50 relevant docs
                # Filter by similarity score
                for doc in docs:
                    if doc[1] >= similarity_threshold: # Check doc score
                        retrieved_count += 1
                        retrieved_docs.append(
                            {
                                "goal": i,
                                "sub_target": defn,
                                "page_content": doc[0].page_content
                            }
                        )
            
                if not retrieved_count:
                    logger.info(f"No Retrieved docs for \nGoal - {i}, \nSub-Target - {defn}\n")
                    logger.info("*"*100)
                else:
                    logger.info(f"Retrieved {retrieved_count} docs for Goal - {i}, \nSub-Target - {defn}\n")
                    logger.info("*"*100)


        return retrieved_docs
                    


if __name__ == '__main__':

    # Initialize Weaviate Client locally via Docker and Initialize OpenAI Embeddings
    weaviate_client = weaviate.connect_to_local()
    openai_embeddings = OpenAIEmbeddings()
    
    doc_retriever = Retrieve(embeddings=openai_embeddings, weaviate_client=weaviate_client, targets_path=os.path.join("results", "targets.json"))
    txt_pth = os.path.join("data", "txts", "14.BASF_$42.93 B_Industrials", "2023", "results.txt")
    db = doc_retriever.get_db(txt_pth)

    retrieved_docs = doc_retriever.retrieve_docs(db=db, similarity_threshold=0.75)

    # Close client connection
    weaviate_client.close()


