from src.data_loader import load_json
from src.indexing import create_documents, build_embeddings

def main():
    # 1. Загрузка данных
    
    articles = load_json("data/articles.json")
    questions = load_json("data/questions.json")
    answers = load_json("data/ground_truth.json")
    
    # 2. Подготовка документов и эмбеддингов
    documents = create_documents(articles)
    embeddings = build_embeddings(documents)
    
    # 3. Генерация plain и RAG-ответов
    
    # 4. Оценка результатов
    
    # 5. Сохранение результатов

if __name__ == "__main__":
    main()