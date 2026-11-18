
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

from evaluation import (
    DEFAULT_DISTANCE_THRESHOLD,
    evaluate_retrieval,
    filter_relevant_chunks,
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
    "В предоставленных документах недостаточно "
    "информации для ответа на этот вопрос."
)


def get_llm_client() -> OpenAI:
    """Создаёт клиент для Groq API."""

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "Set GROQ_API_KEY in your .env file"
        )

    return OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL
    )


def generate_answer(question: str, context: str) -> str:
    """Генерирует ответ только по найденному контексту."""

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
    top_k: int = 3,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
) -> dict:
    """Выполняет RAG с фильтрацией слабого контекста."""

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    logger.info("Question: %s", question)

    # Step 1. Retrieval
    chunks = retrieve_chunks(
        query=question,
        top_k=top_k
    )

    # Step 2. Evaluate retrieval
    evaluation = evaluate_retrieval(
        chunks=chunks,
        distance_threshold=distance_threshold
    )

    logger.info(
        "Retrieval: total=%s, accepted=%s, best_distance=%s",
        evaluation.retrieved_count,
        evaluation.accepted_count,
        evaluation.best_distance
    )

    # Step 3. Reject weak context
    if not evaluation.is_relevant:
        logger.warning("Insufficient context. Skipping LLM call.")

        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "context": "",
            "status": "insufficient_context",
            "retrieval_evaluation": evaluation
        }

    # Step 4. Keep relevant chunks
    relevant_chunks = filter_relevant_chunks(
        chunks=chunks,
        distance_threshold=distance_threshold
    )

    # Step 5. Build context
    context = build_context(relevant_chunks)

    # Step 6. Generate answer
    answer = generate_answer(
        question=question,
        context=context
    )

    # Step 7. Prepare sources
    sources = []

    for index, chunk in enumerate(relevant_chunks, start=1):
        sources.append({
            "reference": index,
            "document_id": chunk["document_id"],
            "source_name": chunk["source_name"],
            "chunk_id": chunk["chunk_id"],
            "distance": chunk["distance"]
        })

    # Step 8. Return result
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context": context,
        "status": "answered",
        "retrieval_evaluation": evaluation
    }



def main():
    """Интерактивный запуск RAG."""

    print("=" * 60)
    print("RAG QUESTION ANSWERING SYSTEM")
    print("=" * 60)

    question = input("\nEnter your question: ").strip()

    result = rag_pipeline(
        question=question,
        top_k=3,
        distance_threshold=0.5
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    print(f"\nStatus: {result['status']}")

    evaluation = result["retrieval_evaluation"]

    print(f"Retrieved chunks: {evaluation.retrieved_count}")
    print(f"Accepted chunks: {evaluation.accepted_count}")
    print(f"Best distance: {evaluation.best_distance}")

    print("\nSOURCES")

    for source in result["sources"]:
        print(
            f"[{source['reference']}] "
            f"{source['source_name']} | "
            f"Chunk: {source['chunk_id']} | "
            f"Distance: {source['distance']:.4f}"
        )


if __name__ == "__main__":
    main()
