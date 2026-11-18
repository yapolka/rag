from chunking import split_text, chunk_documents


def test_short_text():
    text = "Hello world"

    chunks = split_text(
        text,
        chunk_size=100,
        chunk_overlap=20
    )

    assert chunks == ["Hello world"]


def test_overlap():
    text = "ABCDEFGHIJ"

    chunks = split_text(
        text,
        chunk_size=6,
        chunk_overlap=2
    )

    assert chunks == ["ABCDEF", "EFGHIJ"]


def test_empty_text():
    chunks = split_text(
        "",
        chunk_size=100,
        chunk_overlap=20
    )

    assert chunks == []


def test_invalid_overlap():
    try:
        split_text(
            "Hello",
            chunk_size=10,
            chunk_overlap=10
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_chunk_metadata():
    documents = [
        {
            "document_id": "doc_1",
            "source_name": "test.txt",
            "text": "ABCDEFGHIJ"
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=6,
        chunk_overlap=2
    )

    assert len(chunks) == 2

    assert chunks[0]["document_id"] == "doc_1"
    assert chunks[0]["source_name"] == "test.txt"
    assert chunks[0]["chunk_id"] == 0

    assert chunks[1]["chunk_id"] == 1


if __name__ == "__main__":
    test_short_text()
    test_overlap()
    test_empty_text()
    test_invalid_overlap()
    test_chunk_metadata()

    print("All tests passed!")