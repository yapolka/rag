from dataclasses import dataclass


DEFAULT_DISTANCE_THRESHOLD = 0.5


@dataclass
class RetrievalEvaluation:
    """Результат оценки найденного контекста."""

    is_relevant: bool
    best_distance: float | None
    retrieved_count: int
    accepted_count: int


def filter_relevant_chunks(
    chunks: list[dict],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
) -> list[dict]:
    """Оставляет только чанки с допустимым расстоянием."""

    if not 0 <= distance_threshold <= 2:
        raise ValueError(
            "Cosine distance threshold must be between 0 and 2"
        )

    relevant_chunks = []

    for chunk in chunks:
        distance = chunk["distance"]

        if distance is None:
            continue

        if distance <= distance_threshold:
            relevant_chunks.append(chunk)

    return relevant_chunks


def evaluate_retrieval(
    chunks: list[dict],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
) -> RetrievalEvaluation:
    """Оценивает retrieval на основе cosine distance."""

    relevant_chunks = filter_relevant_chunks(
        chunks=chunks,
        distance_threshold=distance_threshold
    )

    distances = [
        chunk["distance"]
        for chunk in chunks
        if chunk.get("distance") is not None
    ]

    best_distance = min(distances) if distances else None

    return RetrievalEvaluation(
        is_relevant=len(relevant_chunks) > 0,
        best_distance=best_distance,
        retrieved_count=len(chunks),
        accepted_count=len(relevant_chunks)
    )
