from retriever import retrieve_chunks
from evaluation import evaluate_retrieval


TEST_QUESTIONS = [
    {
        "question": "What is machine learning?",
        "expected_answerable": True,
    },
    {
        "question": "What is supervised learning?",
        "expected_answerable": True,
    },
    {
        "question": "What is a neural network?",
        "expected_answerable": True,
    },
    {
        "question": "What is the weather on Mars?",
        "expected_answerable": False,
    },
    {
        "question": "Who won the football World Cup in 1930?",
        "expected_answerable": False,
    },
]


THRESHOLDS = [0.3, 0.4, 0.5, 0.6]


def main():

    # Получаем результаты поиска один раз для каждого вопроса.
    retrieved_results = []

    for test in TEST_QUESTIONS:
        chunks = retrieve_chunks(
            query=test["question"],
            top_k=3
        )

        retrieved_results.append((test, chunks))

    # Сравниваем разные пороги.
    for threshold in THRESHOLDS:

        print("\n" + "=" * 70)
        print(f"DISTANCE THRESHOLD: {threshold}")
        print("=" * 70)

        correct = 0

        for test, chunks in retrieved_results:

            evaluation = evaluate_retrieval(
                chunks=chunks,
                distance_threshold=threshold
            )

            predicted_answerable = evaluation.is_relevant

            is_correct = (
                predicted_answerable == test["expected_answerable"]
            )

            if is_correct:
                correct += 1

            print(f"\nQuestion: {test['question']}")
            print(f"Expected answerable: {test['expected_answerable']}")
            print(f"Predicted answerable: {predicted_answerable}")
            print(f"Best distance: {evaluation.best_distance}")
            print(f"Accepted chunks: {evaluation.accepted_count}")
            print(f"Correct prediction: {is_correct}")

        accuracy = correct / len(TEST_QUESTIONS)

        print(f"\nAnswerability accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()