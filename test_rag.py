from rag_pipeline import rag_pipeline


questions = [
    "What is machine learning?",
    "What is supervised learning?",
    "What is a neural network?",
    "What is natural language processing?",
    "How does artificial intelligence work?",
]


def main():
    for question in questions:

        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")

        result = rag_pipeline(
            question=question,
            top_k=3
        )

        print(f"\nANSWER:\n{result['answer']}")

        print("\nSOURCES:")

        for source in result["sources"]:
            print(
                f"- {source['source_name']} "
                f"(chunk {source['chunk_id']})"
            )


if __name__ == "__main__":
    main()