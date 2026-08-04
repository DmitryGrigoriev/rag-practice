from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def create_documents(articles: list[dict]) -> list[Document]:
    """
    Преобразует статьи в объекты Document для последующей индексации.
    Args:
        articles: Список статей, загруженных из JSON.
    Returns:
        Список объектов Document
    """
    return [
        Document(
            page_content=article["text"],
            metadata={
                "id": article["id"],
                "title": article["title"],
            },
        )
        for article in articles
    ]
    
def build_embeddings(
    documents: list[Document],
    model: str = "qwen3-embedding:0.6b"
) -> list[list[float]]:
    """
    Вычисляет эмбеддинги текстового содержимого документов.
    Args:
        documents: Список объектов Document.
        model: Название модели эмбеддингов Ollama.
    Returns:
        Список эмбеддингов, соответствующих входным документам.
    """
    
    embeddings = OllamaEmbeddings(model=model)
    article_embeddings = embeddings.embed_documents([doc.page_content for doc in documents])
    return article_embeddings