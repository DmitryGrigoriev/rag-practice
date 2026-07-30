import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Функция возвращает меру косинусной близости между двумя текстами
    Изменяется от -1 до 1
    1 - два текста идентичны по смыслу
    -1 - два текста противоположны по смыслу
    0 - тексты не связаны друг с другом
    Args:
        a: list[float] - эмбеддинг первого текста
        b: list[float] - эмбеддинг второго текста
    Return:
        Косинусное сходство векторов
    """
    
    cosine_dot = np.dot(a, b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    return cosine_dot / (a_norm * b_norm)