
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from retriever import retrieve_chunks
from prompts import (
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
)


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

FALLBACK_ANSWER = (
    "В предоставленных документах недостаточно информации "
    "для ответа на этот вопрос."
)


def get_llm_client() -> OpenAI:
    """Создаёт клиент для Groq API."""

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "API key is missing. "
            "Set GROQ_API_KEY or OPENAI_API_KEY in .env"
        )

    return OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL
    )


def generate_answer(question: str, context: str) -> str:
    """Генерирует ответ на основе найденного контекста."""

    client = get_llm_client()

    user_prompt = build_user_prompt(
        question=question,
        context=context
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=600
    )

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        return FALLBACK_ANSWER

    return answer.strip()


def rag_pipeline(
    question: str,
    top_k: int = 3
) -> dict:
    """Выполняет полный RAG pipeline."""

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    logger.info("Received question: %s", question)

    # Step 1. Retrieve relevant chunks
    chunks = retrieve_chunks(
        query=question,
        top_k=top_k
    )

    # Step 2. Handle empty retrieval
    if not chunks:
        logger.warning("No relevant chunks retrieved")

        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "context": ""
        }

    # Step 3. Build context
    context = build_context(chunks)

    logger.info("Context created from %s chunks", len(chunks))

    # Step 4. Generate answer
    answer = generate_answer(
        question=question,
        context=context
    )

    # Step 5. Prepare sources
    sources = []

    for index, chunk in enumerate(chunks, start=1):

        source = {
            "reference": index,
            "document_id": chunk["document_id"],
            "source_name": chunk["source_name"],
            "chunk_id": chunk["chunk_id"],
            "distance": chunk["distance"],
        }

        sources.append(source)

    # Step 6. Return structured result
    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context": context
    }

    logger.info("RAG pipeline completed")

    return result


def main():
    """Запускает интерактивный RAG."""

    print("=" * 60)
    print("RAG QUESTION ANSWERING SYSTEM")
    print("=" * 60)

    question = input("\nEnter your question: ").strip()

    result = rag_pipeline(
        question=question,
        top_k=3
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in result["sources"]:

        print(
            f"[{source['reference']}] "
            f"{source['source_name']} | "
            f"Chunk: {source['chunk_id']} | "
            f"Distance: {source['distance']:.4f}"
        )


if __name__ == "__main__":
    main()
