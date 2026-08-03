from typing import Literal

from ollama import chat
from pydantic import BaseModel

class JudgeLabel(BaseModel):
    label: Literal["correct", "partial", "incorrect"]


def judge_answer(
    question: str,
    context: str,
    reference_answer: str,
    candidate_answer: str,
) -> JudgeLabel:
    """
    Оценивает фактическую корректность и полноту ответа кандидата относительно эталона.
    Args:
        question: текст вопроса пользователя
        context: текстовый контекст, сформированный на основе документов, найденных ретривером.
        reference_answer: эталонный ответ
        candidate_answer: ответ модели кандидата
    Returns:
        Возвращает метку соответствия ответа кандидата эталонному ответу.
    """

    prompt = """
    Ты оцениваешь ответ системы на вопрос.
    
    Оцени только фактическую корректность и полноту
    
    Метки:
    correct - ответ передает все существенные факты эталона;
    partial - основная информация верна, но отсутствует существенная часть;
    incorrect - есть фактическая ошибка, противоречие или ответ не отвечает на вопрос.
    
    Краткость и другая формулировка не является ошибкой.
    Не требуй дословного совпадения с эталоном.
    Если ответ содержит все существенные факты, но сформулирован иначе, оцени его как correct
    
    Перед выбором метки отдельно выдели существенные факты эталонного ответа
    и проверь наличие каждого из них в ответе кандидата.

    Дата и время считаются отдельными существенными фактами.
    Если эталон содержит дату и время, а кандидат указал только дату
    или только время, ответ должен получить метку partial.
    
    Верни только JSON с полем label.
    """
    
    response = chat(
        model="qwen3:4b-instruct",
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": f"""Вопрос:\n{question}\n
                Контекст: \n{context}\n
                Эталонный ответ:\n{reference_answer}\n
                Ответ кандидата:\n{candidate_answer}\n
                """
                
            }
        ],
        format=JudgeLabel.model_json_schema(),
        options={
            "temperature": 0,
            "num_predict": 30,
            "num_ctx": 4096
        }
    )
    
    return JudgeLabel.model_validate_json(
        response["message"]["content"]
    )