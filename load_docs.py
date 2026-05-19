from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader


DOCS_PATH = Path("data/docs/external/docker-docs/content/get-started")


def load_markdown_docs():
    documents = []

    for pattern in ("**/*.md", "**/*.mdx"):
        loader = DirectoryLoader(
            str(DOCS_PATH),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        documents.extend(loader.load())

    return documents


if __name__ == "__main__":
    documents = load_markdown_docs()

    print(f"Loaded documents: {len(documents)}")

    if documents:
        first_doc = documents[0]
        print("\nFirst document metadata:")
        print(first_doc.metadata)
        print("\nFirst 500 characters:")
        print(first_doc.page_content[:500])
