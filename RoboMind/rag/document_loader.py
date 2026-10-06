from pathlib import Path
from rag.knowledge_base import add_document


DOCUMENT_FOLDER = Path("rag/documents")


def load_documents():

    for file in DOCUMENT_FOLDER.glob("*.txt"):

        text = file.read_text(
            encoding="utf-8"
        )

        add_document(
            text=text,
            source=file.name
        )

        print(f"Loaded: {file.name}")


if __name__ == "__main__":
    load_documents()