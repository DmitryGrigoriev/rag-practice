import evaluate

def extract_answers(data: list[dict[str, str]], answer_field: str) -> tuple[list[str], list[str]]:
    """
    Формирует списки предсказаний модели и эталонных ответов для вычисления метрик.
    Args:
        data: dict - json с ответами LLM и RAG
        answer_field: str - поле эталонного ответа и модели кандидата
    Return:
        tuple: - кортеж из списков эталонных ответов и ответов кандидатов
    """
    
    predictions = []
    references = []

    for item in data:
        predictions.append(item[answer_field])
        references.append(item["ground_truth_answer"])

    return predictions, references

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Функция для подсчета BLEU
    Args:
        predictions: list[str] - список эталонных ответов
        references : list[str] - список ответов кандидатов
    Return:
        float: - подсчитанное bleu
    """
    bleu = evaluate.load('bleu')
    references = [[reference] for reference in references]
    return bleu.compute(
        predictions=predictions,
        references=references
    )['bleu']

def compute_rouge(predictions: list[str], references: list[str]) -> float:
    """
    Функция для подсчета ROUGE-L
    Args:
        predictions: list[str] - список эталонных ответов
        references : list[str] - список ответов кандидатов
    Return:
        float: - подсчитанное ROUGE-L
    """
    rouge = evaluate.load('rouge')
    return rouge.compute(
        predictions=predictions,
        references=references
    )['rougeL']

def compute_recall_at_k(results: list[dict], k: int = 3) -> float:
    """
    Вычисляет Recall@k для retrieval.
    Args:
        results: результаты retrieval с relevant_document_id
                и списком retrieved документов.
        k: количество первых retrieved документов для проверки.
    Returns:
        Доля вопросов, для которых релевантный документ попал в top-k.
    """
    recall_k = 0
    
    for result in results:
        relevant_doc = result['relevant_document_id'] 
        for retrieved_doc  in result['retrieved'][:k]:
            if retrieved_doc['document_id'] == relevant_doc:
                recall_k += 1
                break
    
    return recall_k / len(results)

def compute_label_distribution(results: list[dict]) -> dict[str, float]:
    """
    Вычисляем распределение меток LLM as judge
    Args:
        results: результаты RAG с метками LLM as judge.
    Return:
        Возвращает словарь с распределением меток
    """
    
    total = len(results)
    correct, partial, incorrect = 0, 0, 0
    
    for result in results:
        if result['judge_label'] == "correct":
            correct += 1
        elif result['judge_label'] == 'partial':
            partial += 1
        elif result["judge_label"] == "incorrect":
            incorrect += 1
        else:
            raise ValueError("Неизвестная метка")
            
    return {
        "correct": correct / total,
        "partial": partial / total,
        "incorrect": incorrect / total
    }
    
def compute_human_vs_judge_agreement(results: list[dict]) -> dict[str, float]:
    """
    Вычисляем соответствие меток LLM as judge и вручную размеченными метками
    Args:
        results: результаты RAG с метками LLM as judge и human label
    Return:
        Возвращает долю совпадений меток human label и LLM as judge
    """
    
    total = len(results)
    matches = 0
    
    for result in results:
        if result['judge_label'] == result['human_label']:
            matches += 1
            
    return {
        "agreement": matches / total
    }
    