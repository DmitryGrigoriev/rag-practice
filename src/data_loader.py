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
    