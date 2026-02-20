from src.ingest.schema import Document, Metadata

def test_document_creation():
    meta = Metadata(source="test.md", language="en")
    doc = Document(
        title="Test Doc",
        content="This is a test document.",
        metadata=meta
    )

    assert doc.doc_id is not None
    assert doc.metadata.source == "test.md"
    