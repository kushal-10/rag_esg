import os
import json
import weaviate
from tqdm import tqdm
from typing import Dict
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore


class Retrieve():

    """
    Extract Relevant passages for each sub-target for a particular PDF
    """

    def __init__(self, embeddings = OpenAIEmbeddings(), weaviate_client = None, targets_path: str = os.path.join("data", "targets.json")):
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
    
    # @staticmethod
    # def retriever_query():
    #     pass

    # @staticmethod
    # def classification_query():
    #     pass
    
    def retrieve_docs(self, db: WeaviateVectorStore = None) -> Dict:
        """
        Return a Dict object containing T/F Classifications of each Sub-Target for a given annual report

        Args:
            db: Weaviate Database based on sepearate documents from a given text file 
        
        Returns:
            classifications: A dict containing classifications of each Sub-Target
        """

        query = "Does the company have a policy to improve employee health & safety?"
        docs = db.similarity_search_with_score(query, k=5)


if __name__ == '__main__':

    # Initialize Weaviate Client locally via Docker and Initialize OpenAI Embeddings
    weaviate_client = weaviate.connect_to_local()
    openai_embeddings = OpenAIEmbeddings()
    
    classifier = Retrieve(embeddings=openai_embeddings, weaviate_client=weaviate_client, targets_path=os.path.join("data", "targets.json"))
    txt_pth = os.path.join("data", "txts", "14.BASF_$42.93 B_Industrials", "2023", "results.txt")
    db = classifier.get_db(txt_pth)

    # Close client connection
    weaviate_client.close()


# for doc in docs:
#     print(f"{doc[1]:.3f}", ":", doc[0].page_content)


