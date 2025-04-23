from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore

loader = TextLoader("data/txts/14.BASF_$42.93 B_Industrials/2023/results.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

weaviate_client = weaviate.connect_to_local()
db = WeaviateVectorStore.from_documents(docs, embeddings, client=weaviate_client)

query = "By 2030, ensure sustainable food production systems and implement resilient agricultural practices that increase productivity and production, that help maintain ecosystems, that strengthen capacity for adaptation to climate change, extreme weather, drought, flooding and other disasters and that progressively improve land and soil quality"
docs = db.similarity_search_with_score("country", k=50)

for doc in docs:
    print(f"{doc[1]:.3f}", ":", doc[0].page_content[:100] + "...")

weaviate_client.close()
