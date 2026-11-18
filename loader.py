from pathlib import Path


def clean_text(text: str) -> str:
    """Очищает текст от лишних пробелов и пустых строк."""

    lines = text.splitlines()

    cleaned_lines = [
        " ".join(line.split())
        for line in lines
        if line.strip()
    ]

    return "\n".join(cleaned_lines)


def load_documents(directory: str = "knowledge_base") -> list[dict]:
    """Загружает и очищает текстовые документы из папки."""

    documents = []
    folder = Path(directory)

    if not folder.exists():
        raise FileNotFoundError(
            f"Папка '{directory}' не найдена."
        )

    for file_path in sorted(folder.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        cleaned_text = clean_text(text)

        if not cleaned_text:
            continue

        document = {
            "document_id": file_path.stem,
            "source_name": file_path.name,
            "text": cleaned_text,
        }

        documents.append(document)

    return documents


def main():
    """Запускает загрузку документов и выводит результаты."""

    documents = load_documents()

    print(f"Загружено документов: {len(documents)}")

    for document in documents:
        print("-" * 50)
        print(f"Document ID: {document['document_id']}")
        print(f"Source: {document['source_name']}")
        print(f"Length: {len(document['text'])} characters")
        print(f"Preview: {document['text'][:100]}")


if __name__ == "__main__":
    main()