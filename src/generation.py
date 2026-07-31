from ollama import chat


def answer_without_rag(question: str) -> str:
    """
    Генерирует ответ на вопрос с помощью LLM без использования внешнего контекста.
    Args:
        question: Текст вопроса пользователя.
    Returns:
        Текст ответа, сгенерированного моделью.
    """
    response = chat(
        model='qwen3:4b-instruct',
        messages=[{
            "role": "system",
            "content": (
                "Отвечай на вопросы кратко и по существу. "
                "Если не знаешь ответа, ответь не знаю. "
                "Не выдумывай факты."
            )
        },                   
        {
            "role": "user",
            "content": question
        }        
        ],
        options={
            "temperature": 0
        },
        think=False
    )
    
    return response["message"]["content"]