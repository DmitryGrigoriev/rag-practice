from src.data_loader import load_json, merge_questions_and_answers
from src.indexing import create_documents, build_embeddings

def main():
    # 1. Загрузка данных
    
    articles = load_json("data/articles.json")
    questions = load_json("data/questions.json")
    answers = load_json("data/ground_truth.json")
    
    question_answer = merge_questions_and_answers(questions, answers)
    
    # 2. Подготовка документов и эмбеддингов
    documents = create_documents(articles)
    embeddings = build_embeddings(documents)
    
    # 3. Генерация plain и RAG-ответов
    
    # 4. Оценка результатов
    
    # 5. Сохранение результатов
    
    return question_answer

if __name__ == "__main__":
    main()