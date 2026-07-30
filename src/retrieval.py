import numpy as np

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Вычисляет косинусное сходство между двумя векторами эмбеддингов.
    
    Args:
        a: Эмбеддинг первого текста
        b: Эмбеддинг второго текста

    Returns:
        Косинусное сходство векторов
    """
    
    cosine_dot = np.dot(a, b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    return cosine_dot / (a_norm * b_norm)

def retrieve_documents(
    query: str,
    documents: list[Document],
    article_embeddings: list[list[float]],
    top_k: int = 3,
    model: str='qwen3-embedding:0.6b',
):
    """
    Находит top-k наиболее релевантных документов, сравнивая эмбеддинг запроса с эмбеддингами документов по косинусному сходству.
    Args:
        query: запрос
        documents: проиндексированные документы
        article_embedding: эмбеддинги статей
        top_k: топ-к найболее релевантных к запросу документов
    Return:
        Список словарей с найденным документом и значением косинусного сходства.
        [
    {
        "document": ...,
        "score": ...
    }
]
    """
    
    embeddings = OllamaEmbeddings(model=model) 
    query_embeddings = embeddings.embed_query(query)
    
    scores = np.array([
        cosine_similarity(query_embeddings, article_embedding)
        for article_embedding in article_embeddings
    ])
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return [
        {
            "document": documents[idx],
            "score": float(scores[idx])
        }
        for idx in top_indices
    ]