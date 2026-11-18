import logging

import weaviate

from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
)
from weaviate.util import generate_uuid5


logger = logging.getLogger(__name__)

COLLECTION_NAME = "KnowledgeChunk"


def connect_to_weaviate():
    """Подключается к локальному Weaviate."""

    client = weaviate.connect_to_local(
        host="localhost",
        port=8081,
        grpc_port=50051
    )

    if not client.is_ready():
        client.close()
        raise ConnectionError("Weaviate is not ready")

    logger.info("Connected to Weaviate")

    return client


def create_collection(client):
    """Создаёт коллекцию для хранения чанков."""

    if client.collections.exists(COLLECTION_NAME):
        logger.info("Collection already exists: %s", COLLECTION_NAME)
        return client.collections.use(COLLECTION_NAME)

    collection = client.collections.create(
        name=COLLECTION_NAME,

        vector_config=Configure.Vectors.self_provided(),

        properties=[
            Property(
                name="document_id",
                data_type=DataType.TEXT
            ),
            Property(
                name="source_name",
                data_type=DataType.TEXT
            ),
            Property(
                name="chunk_id",
                data_type=DataType.INT
            ),
            Property(
                name="text",
                data_type=DataType.TEXT
            ),
        ]
    )

    logger.info("Collection created: %s", COLLECTION_NAME)

    return collection


def insert_chunks(
    collection,
    chunks: list[dict],
    embedder
) -> tuple[int, int]:
    """Создаёт embeddings и сохраняет чанки без дублирования."""

    inserted = 0
    skipped = 0

    for chunk in chunks:

        chunk_uuid = generate_uuid5({
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
        })

        existing = collection.query.fetch_object_by_id(
            chunk_uuid
        )

        if existing is not None:
            skipped += 1
            continue

        embedding = embedder(chunk["text"])

        collection.data.insert(
            properties={
                "document_id": chunk["document_id"],
                "source_name": chunk["source_name"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            },
            vector=embedding,
            uuid=chunk_uuid
        )

        inserted += 1

        logger.info(
            "Inserted chunk %s from %s",
            chunk["chunk_id"],
            chunk["source_name"]
        )

    logger.info(
        "Insertion completed: inserted=%s, skipped=%s",
        inserted,
        skipped
    )

    return inserted, skipped


def count_objects(collection) -> int:
    """Возвращает количество объектов в коллекции."""

    result = collection.aggregate.over_all(
        total_count=True
    )

    return result.total_count


def main():
    """Проверяет подключение и состояние коллекции."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    client = connect_to_weaviate()

    try:
        collection = create_collection(client)

        total_objects = count_objects(collection)

        logger.info("Total objects: %s", total_objects)

    finally:
        client.close()
        logger.info("Weaviate connection closed")


if __name__ == "__main__":
    main()