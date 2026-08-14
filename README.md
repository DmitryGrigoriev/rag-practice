# RAG Pipeline Evaluation

Учебный проект по созданию и оценке RAG-системы на небольшом наборе документов.

## Цель проекта

Сравнить качество ответов базовой LLM и RAG-подхода, а также оценить качество retrieval и генерации с помощью автоматических метрик, ручной разметки и LLM-as-a-Judge.

## Стек

- Python
- Jupyter Notebook
- Ollama
- qwen3:4b-instruct
- qwen3-embedding

## Структура проекта

```text
.
├── data
│   ├── articles.json
│   ├── questions.json
│   ├── answers.json
│   └── llm_answers_labeled.json
├── notebooks
│   ├── eda.ipynb
│   └── rag_evaluation.ipynb
├── src
│   ├── data_loader.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── judge.py
│   ├── evaluation.py
│   ├── save_json.py
│   └── main.py
└── README.md
```

## Pipeline

1. Загрузка документов.
2. Создание объектов `Document`.
3. Построение эмбеддингов.
4. Retrieval релевантных документов.
5. Формирование контекста.
6. Генерация ответа без RAG.
7. Генерация ответа с RAG.
8. Оценка результатов.

## Метрики

### Generation

| Метрика | Plain LLM | RAG |
| --- | --- | --- |
| BLEU | 0.01 | 0.31 |
| ROUGE | 0.02 | 0.82 |

### Retrieval

| Метрика | Значение |
| --- | --- |
| Recall@1 | 0.92 |
| Recall@3 | 1.00 |

### LLM-as-a-Judge

| Метка | Доля |
| --- | --- |
| Correct | 94% |
| Partial | 4% |
| Incorrect | 2% |

### Human vs Judge

| Метрика | Значение |
| --- | --- |
| Agreement | 98% |

## Запуск

```bash
python -m src.main
```

## Выводы

- RAG существенно улучшил качество ответов по сравнению с базовой моделью.
- Retrieval показал высокое качество: релевантный документ попадал в top-3 в 100% случаев.
- LLM-as-a-Judge продемонстрировал высокое согласие с ручной разметкой.