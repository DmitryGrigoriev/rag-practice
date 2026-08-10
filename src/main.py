import json
import random
from pathlib import Path
from tqdm import tqdm

from src.data_loader import load_json, merge_questions_and_answers
from src.indexing import create_documents, build_embeddings
from src.generation import answer_without_rag, answer_with_rag
from src.retrieval import retrieve_documents, build_context
from src.evaluation import extract_answers, compute_bleu, compute_rouge
from src.judge import judge_answer
from src.save_json import save_json

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
    llm_answers = []
    for (i, item) in enumerate(
            tqdm(question_answer, desc='Получение ответа'),
            start=1
        ):
        
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
            
            llm_answers.append({
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
    
    llm_predictions, references = extract_answers(llm_answers, "llm_answer")
    rag_predictions, _ = extract_answers(llm_answers, "rag_answer")
    
    metrics = {
        "llm": {
            "bleu": compute_bleu(llm_predictions, references),
            "rouge-L": compute_rouge(llm_predictions, references)
        },
        "rag": {
            "bleu": compute_bleu(rag_predictions, references),
            "rouge-L": compute_rouge(rag_predictions, references)
        }
    }
    
    for item in tqdm(llm_answers, desc="Оценка ответов"):
        judgement = judge_answer(
            question=item['question'],
            context=item['context'],
            reference_answer=item['ground_truth_answer'],
            candidate_answer=item['rag_answer']
        )
        item['judge_label'] = judgement.label

    # 5. Сохранение результатов
    save_json('data/llm_answers.json', llm_answers)
    save_json('data/metrics.json', metrics)

if __name__ == "__main__":
    main()