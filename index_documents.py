import logging

from loader import load_documents
from chunking import chunk_documents
from embedder import create_embedding

from weaviate_store import (
    connect_to_weaviate,
    create_collection,
    insert_chunks,
    count_objects,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Запускает полный pipeline индексации документов."""

    logger.info("Starting RAG indexing pipeline")

    # Step 1. Load documents
    documents = load_documents()

    logger.info("Loaded documents: %s", len(documents))

    if not documents:
        raise ValueError("No documents found")

    # Step 2. Split documents into chunks
    chunks = chunk_documents(
        documents=documents,
        chunk_size=500,
        chunk_overlap=100
    )

    logger.info("Created chunks: %s", len(chunks))

    if not chunks:
        raise ValueError("No chunks created")

    # Step 3. Connect to Weaviate
    client = connect_to_weaviate()

    try:
        # Step 4. Create or get collection
        collection = create_collection(client)

        # Step 5. Generate embeddings and insert chunks
        inserted, skipped = insert_chunks(
            collection=collection,
            chunks=chunks,
            embedder=create_embedding
        )

        # Step 6. Verify stored objects
        total_objects = count_objects(collection)

        logger.info("Inserted chunks: %s", inserted)
        logger.info("Skipped chunks: %s", skipped)
        logger.info("Objects in Weaviate: %s", total_objects)

        if total_objects < len(chunks):
            raise RuntimeError(
                "Fewer objects stored than expected"
            )

        logger.info("Indexing completed successfully")

    finally:
        client.close()
        logger.info("Weaviate connection closed")


if __name__ == "__main__":
    main()