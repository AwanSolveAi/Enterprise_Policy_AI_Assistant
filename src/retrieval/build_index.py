import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/policy_chunks.json"
)

OUTPUT_DIR = Path(
    "data/vector_store"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INDEX_FILE = (
    OUTPUT_DIR /
    "policy_faiss.index"
)

METADATA_FILE = (
    OUTPUT_DIR /
    "policy_metadata.json"
)


# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


# Number of chunks encoded at once
BATCH_SIZE = 32


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:

        raise ValueError(
            "No chunks found in input file."
        )

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        "\nThe model may download the first time."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "\nEmbedding model loaded successfully."
    )

    return model


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    model,
    texts
):

    print("\n" + "=" * 70)
    print("CREATING EMBEDDINGS")
    print("=" * 70)

    print(
        f"\nTexts to encode: "
        f"{len(texts)}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    return embeddings


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(
    embeddings
):

    print("\n" + "=" * 70)
    print("BUILDING FAISS INDEX")
    print("=" * 70)

    dimension = embeddings.shape[1]

    print(
        f"\nEmbedding dimension: "
        f"{dimension}"
    )

    print(
        f"Number of vectors: "
        f"{embeddings.shape[0]}"
    )


    # --------------------------------------------------------
    # Inner Product + normalized vectors
    #
    # For normalized embeddings:
    #
    # cosine similarity = inner product
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        "\nFAISS index created successfully."
    )

    print(
        f"Vectors stored: "
        f"{index.ntotal}"
    )

    return index


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(
    index,
    chunks
):

    print("\n" + "=" * 70)
    print("SAVING VECTOR STORE")
    print("=" * 70)


    faiss.write_index(
        index,
        str(INDEX_FILE)
    )


    # --------------------------------------------------------
    # Store metadata separately
    # --------------------------------------------------------

    metadata = []


    for chunk in chunks:

        metadata.append({

            "chunk_id":
                chunk.get(
                    "chunk_id"
                ),

            "document_id":
                chunk.get(
                    "document_id"
                ),

            "title":
                chunk.get(
                    "title"
                ),

            "category":
                chunk.get(
                    "category"
                ),

            "country":
                chunk.get(
                    "country"
                ),

            "policy_type":
                chunk.get(
                    "policy_type"
                ),

            "source":
                chunk.get(
                    "source"
                ),

            "source_url":
                chunk.get(
                    "source_url"
                ),

            "chunk_index":
                chunk.get(
                    "chunk_index"
                ),

            "chunk_text":
                chunk.get(
                    "chunk_text"
                ),

        })


    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )


    print(
        "\nFAISS index saved to:"
    )

    print(
        INDEX_FILE
    )


    print(
        "\nMetadata saved to:"
    )

    print(
        METADATA_FILE
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_index(
    index,
    chunks
):

    print("\n" + "=" * 70)
    print("VALIDATING INDEX")
    print("=" * 70)


    if index.ntotal != len(chunks):

        raise ValueError(
            "FAISS vector count does not "
            "match chunk count."
        )


    print(
        "\nValidation successful."
    )

    print(
        f"Chunks: "
        f"{len(chunks)}"
    )

    print(
        f"Vectors: "
        f"{index.ntotal}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GITLAB HR POLICY EMBEDDING PIPELINE")
    print("=" * 70)


    # --------------------------------------------------------
    # Load policy chunks
    # --------------------------------------------------------

    print(
        "\nLoading policy chunks..."
    )

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )


    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    texts = []

    for chunk in chunks:

        text = chunk.get(
            "chunk_text",
            ""
        )

        if not text.strip():

            raise ValueError(
                f"Empty chunk found: "
                f"{chunk.get('chunk_id')}"
            )

        texts.append(
            text
        )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        model,
        texts
    )


    # --------------------------------------------------------
    # Validate embedding shape
    # --------------------------------------------------------

    print(
        "\nEmbedding matrix shape:"
    )

    print(
        embeddings.shape
    )


    if embeddings.shape[0] != len(chunks):

        raise ValueError(
            "Number of embeddings does not "
            "match number of chunks."
        )


    # --------------------------------------------------------
    # Build FAISS
    # --------------------------------------------------------

    index = build_faiss_index(
        embeddings
    )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_index(
        index,
        chunks
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_index(
        index,
        chunks
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("EMBEDDING PIPELINE COMPLETE")
    print("=" * 70)

    print(
        "\nYour vector database is ready."
    )

    print(
        "\nFiles created:"
    )

    print(
        f"1. {INDEX_FILE}"
    )

    print(
        f"2. {METADATA_FILE}"
    )


    print(
        "\nNext step:"
    )

    print(
        "Build the semantic retriever and test "
        "real HR policy questions."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()