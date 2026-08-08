from src.data_loader import load_json, merge_questions_and_answers
from src.indexing import create_documents, build_embeddings
from src.generation import answer_without_rag, answer_with_rag
from src.retrieval import retrieve_documents, build_context

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
    llm_answer = []
    for (i, item) in enumerate(question_answer, start=1):
        
        try:
            
            retrieved = retrieve_documents(
                query=item["question"],
                documents=documents,
                article_embeddings=embeddings,
                top_k=3
            )
            
            context = build_context(retrieved)
            plain_answer = answer_without_rag(item['question'])
            rag_answer = answer_with_rag(item['question'], context)
            
            llm_answer.append({
                "question": item['question'],
                "ground_truth_answer": item['answer'],
                "llm_answer": plain_answer,
                "rag_answer": rag_answer,
                "retrieved": [
                    {"document_id": result["document"].metadata["id"],
                    "score": result['score']
                    }
                    for result in retrieved
                    ],
                "context": context
                }
            )

        except Exception as error:
            print(f"Ошибка на объекте {i}: {error}")
            break
    # 4. Оценка результатов
    
    # 5. Сохранение результатов
    
    return llm_answer

if __name__ == "__main__":
    print(main()[35])