from loader import DocumentLoader

if __name__ == "__main__":
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    print(f"\n✅ Загружено {len(documents)} документов")
    for doc in documents:
        print(f"  - {doc.source_name}: {len(doc.content)} символов")