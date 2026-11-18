from functools import lru_cache

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Загружает embedding-модель один раз."""

    return SentenceTransformer(EMBEDDING_MODEL)


def create_embedding(text: str) -> list[float]:
    """Создаёт embedding документа."""

    if not text.strip():
        raise ValueError("Cannot create embedding for empty text")

    model = get_embedding_model()

    embedding = model.encode(
        f"passage: {text}",
        normalize_embeddings=True
    )

    return embedding.tolist()


def create_query_embedding(query: str) -> list[float]:
    """Создаёт embedding пользовательского вопроса."""

    if not query.strip():
        raise ValueError("Cannot create embedding for empty query")

    model = get_embedding_model()

    embedding = model.encode(
        f"query: {query}",
        normalize_embeddings=True
    )

    return embedding.tolist()


def main():
    text = "Machine learning is a subset of artificial intelligence."

    embedding = create_embedding(text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")


if __name__ == "__main__":
    main()