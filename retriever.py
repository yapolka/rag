import logging

from weaviate.classes.query import MetadataQuery

from embedder import create_query_embedding
from weaviate_store import (
    COLLECTION_NAME,
    connect_to_weaviate,
    count_objects,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def retrieve_chunks(
    query: str,
    top_k: int = 3
) -> list[dict]:
    """Ищет наиболее релевантные чанки в Weaviate."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    client = connect_to_weaviate()

    try:
        if not client.collections.exists(COLLECTION_NAME):
            raise RuntimeError(
                "Collection does not exist. "
                "Run index_documents.py first."
            )

        collection = client.collections.use(COLLECTION_NAME)

        total_objects = count_objects(collection)

        if total_objects == 0:
            raise RuntimeError(
                "Collection is empty. "
                "Run index_documents.py first."
            )

        logger.info("Searching for: %s", query)

        query_vector = create_query_embedding(query)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(
                distance=True
            ),
            return_properties=[
                "document_id",
                "source_name",
                "chunk_id",
                "text",
            ]
        )

        results = []

        for obj in response.objects:

            result = {
                "document_id": obj.properties["document_id"],
                "source_name": obj.properties["source_name"],
                "chunk_id": obj.properties["chunk_id"],
                "text": obj.properties["text"],
                "distance": obj.metadata.distance,
            }

            results.append(result)

        logger.info("Retrieved %s chunks", len(results))

        return results

    finally:
        client.close()


def main():
    """Проверяет semantic retrieval."""

    query = input("Enter your question: ").strip()

    results = retrieve_chunks(
        query=query,
        top_k=3
    )

    print("\nSEARCH RESULTS")
    print("=" * 60)

    for index, chunk in enumerate(results, start=1):

        print(f"\nResult #{index}")
        print("-" * 60)

        print(f"Document ID: {chunk['document_id']}")
        print(f"Source: {chunk['source_name']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Distance: {chunk['distance']:.4f}")

        print("\nText:")
        print(chunk["text"])


if __name__ == "__main__":
    main()