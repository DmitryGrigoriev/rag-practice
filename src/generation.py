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


def answer_with_rag(question: str, context: str) -> str:
    """
    Генерирует ответ LLM на вопрос с использованием контекста из найденных документов.
    Args:
        question: Текст вопроса пользователя.
        context: Текстовый контекст, сформированный на основе документов, найденных ретривером.
    Returns:
        Текст ответа, сгенерированного моделью.
    """
    
    response = chat(
        model='qwen3:4b-instruct',
        messages=[
            {
                "role": "system",
                "content": (
                    "Ответь на вопрос только на основе предоставленных документов. "
                    "Дай краткий и точный ответ. "
                    "Если ответа в документах нет, ответь: «Нет ответа в контексте»."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Документы:\n{context}\n\n"
                    f"Вопрос:\n{question}"
                )
            }
        ],
        options={
            "temperature": 0
        },
        think=False
    )
    return response["message"]["content"]