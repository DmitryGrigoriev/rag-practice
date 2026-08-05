import json
from pathlib import Path

def load_json(path: str | Path) -> list[dict]:
    """
    Функция для считывания json.
    Args:
        path: str | Path - расположение json
    Return:
        list[dict] - список json
    """
    result = Path(path).read_text(encoding='utf-8')
    return json.loads(result)

def merge_questions_and_answers(questions: list[dict], answers: list[dict]) -> list[dict]:
    """
    Объединяем вопросы и ответы в единый словарь
    Args:
        questions: список словарей с вопросами
        answers: спиcок словарей с ответами
    Returns:
        Список словарей, содержащих объединенные данные вопроса и соответствующего эталонного ответа.
    """
    
    answer_index = {answer["id"]: answer for answer in answers}
    
    missing_ids = {
        question["id"]
        for question in questions
        if question["id"] not in answer_index
    }

    if missing_ids:
        raise ValueError(f"Missing answers for ids: {sorted(missing_ids)}")

    question_answer = [
        {**question, **answer_index[question["id"]]}
        for question in questions
    ]

    return question_answer