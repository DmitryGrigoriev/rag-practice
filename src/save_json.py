import json
from pathlib import Path
from typing import Any

def save_json(path: str | Path, data: list[dict[str, any]] | dict[str, Any]) -> None:
    """
    Сохраняет Python-объект в JSON-файл.
    Args:
        path: Путь к выходному JSON-файлу.
        data: Python-объект для сериализации в JSON.
    """
    
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )