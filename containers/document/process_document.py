import json
import os
import re
import time

import mlflow


def main():
    config_path = os.environ.get("CONFIG_PATH", "/amortized/config.json")
    with open(config_path) as f:
        config = json.load(f)

    input_path = config["input_path"]
    filename = config["filename"]
    chunker_type = config.get("chunker_type", "sentence")
    chunk_size = config.get("chunk_size", 2048)
    chunk_overlap = config.get("chunk_overlap", 200)
    tokenizer = config.get("tokenizer", "cl100k_base")

    # Step 1: Convert with docling
    start = time.time()
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(input_path)
    content = result.document.export_to_markdown()
    processing_time = time.time() - start

    print(f"Converted {filename}: {len(content)} chars in {processing_time:.1f}s")

    if not content.strip():
        print("WARNING: No content extracted from document")

    # Step 2: Chunk with chonkie
    chunks = []
    if content.strip():
        from chonkie import SentenceChunker, TokenChunker, RecursiveChunker

        if chunker_type == "token":
            chunker = TokenChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer
            )
        elif chunker_type == "recursive":
            chunker = RecursiveChunker(chunk_size=chunk_size, tokenizer=tokenizer)
        else:
            chunker = SentenceChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer
            )

        raw_chunks = chunker.chunk(content)
        chunks = [{"text": c.text, "token_count": c.token_count} for c in raw_chunks]
        print(f"Chunked into {len(chunks)} chunks using {chunker_type}")

    # Step 3: Upload to MLflow
    mlflow.set_experiment("amortized/documents")
    with mlflow.start_run(run_name=filename) as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "job_type": "document",
            "filename": filename,
            "format": "md",
            "processing_time": str(processing_time),
            "content_length": str(len(content)),
            "chunk_count": str(len(chunks)),
        })

        # Upload source file
        mlflow.log_artifact(input_path, "source")

        # Upload parsed content
        content_path = "/tmp/parsed_content.md"
        with open(content_path, "w") as f:
            f.write(content)
        mlflow.log_artifact(content_path, "")

        # Upload chunks
        if chunks:
            os.makedirs("/tmp/chunks", exist_ok=True)
            heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
            metadata = []
            for i, chunk in enumerate(chunks):
                chunk_path = f"/tmp/chunks/chunk_{i:03d}.md"
                with open(chunk_path, "w") as f:
                    f.write(chunk["text"])
                headings = [m.group(2).strip() for m in heading_re.finditer(chunk["text"])]
                metadata.append({
                    "chunk_index": i,
                    "num_tokens": chunk["token_count"],
                    "headings": headings,
                    "page_numbers": [],
                })

            metadata_path = "/tmp/chunks/metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            mlflow.log_artifacts("/tmp/chunks", "chunks")

    print(f"AMORTIZED_MLFLOW_RUN_ID={run_id}")


if __name__ == "__main__":
    main()
