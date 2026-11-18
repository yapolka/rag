from loader import load_documents


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[str]:
    """Разбивает текст на чанки с перекрытием."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - chunk_overlap

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[dict]:
    """Разбивает документы на чанки и добавляет метаданные."""

    all_chunks = []

    for document in documents:

        text = document["text"]

        text_chunks = split_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for chunk_index, chunk_text in enumerate(text_chunks):

            chunk = {
                "document_id": document["document_id"],
                "source_name": document["source_name"],
                "chunk_id": chunk_index,
                "text": chunk_text,
            }

            all_chunks.append(chunk)

    return all_chunks


def main():
    """Загружает документы, разбивает их на чанки и выводит результат."""

    documents = load_documents()

    chunks = chunk_documents(
        documents=documents,
        chunk_size=500,
        chunk_overlap=100
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks[:5]:
        print("-" * 60)
        print(f"Document ID: {chunk['document_id']}")
        print(f"Source: {chunk['source_name']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Length: {len(chunk['text'])}")
        print(f"Text: {chunk['text']}")


if __name__ == "__main__":
    main()