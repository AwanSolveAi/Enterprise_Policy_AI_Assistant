import json
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/hr_policies.json"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "policy_chunks.json"
)


# Target chunk size in characters
TARGET_CHARS = 1800

# Maximum chunk size
MAX_CHARS = 2400

# Minimum useful chunk size
MIN_CHARS = 300

# Overlap between neighboring chunks
OVERLAP_CHARS = 250


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize whitespace while preserving paragraphs.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Normalize spaces and tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Normalize excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Remove unnecessary whitespace
    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(
        lines
    ).strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):
    """
    Basic sentence splitter.

    This intentionally avoids external NLP dependencies.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# WORD-SAFE SPLITTING
# ============================================================

def split_long_text(
    text,
    max_chars
):
    """
    Split very long text without ever cutting a word.

    This fixes the previous problem where a chunk could
    start with something like:

        icies.

    instead of:

        policies.
    """

    words = text.split()

    chunks = []

    current = ""

    for word in words:

        # Normal case
        if not current:

            current = word
            continue

        proposed = (
            current
            + " "
            + word
        )

        if len(proposed) <= max_chars:

            current = proposed

        else:

            chunks.append(
                current.strip()
            )

            current = word

    if current:

        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# BUILD CHUNKS FROM PARAGRAPH
# ============================================================

def build_chunks_from_paragraph(
    paragraph
):
    """
    Convert one paragraph into retrieval-friendly chunks.
    """

    paragraph = paragraph.strip()

    if not paragraph:
        return []

    # --------------------------------------------------------
    # Short paragraph
    # --------------------------------------------------------

    if len(paragraph) <= MAX_CHARS:

        return [
            paragraph
        ]


    # --------------------------------------------------------
    # Long paragraph
    # --------------------------------------------------------

    sentences = split_sentences(
        paragraph
    )

    chunks = []

    current = ""


    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue


        # ----------------------------------------------------
        # Extremely long sentence
        # ----------------------------------------------------

        if len(sentence) > MAX_CHARS:

            # Save existing chunk first
            if current:

                chunks.append(
                    current.strip()
                )

                current = ""


            # Word-safe splitting
            long_chunks = split_long_text(
                sentence,
                MAX_CHARS
            )

            chunks.extend(
                long_chunks
            )

            continue


        # ----------------------------------------------------
        # Add sentence to current chunk
        # ----------------------------------------------------

        if not current:

            current = sentence

            continue


        proposed = (
            current
            + " "
            + sentence
        )


        if len(proposed) <= TARGET_CHARS:

            current = proposed

        else:

            chunks.append(
                current.strip()
            )

            current = sentence


    # --------------------------------------------------------
    # Remaining text
    # --------------------------------------------------------

    if current:

        chunks.append(
            current.strip()
        )


    return chunks


# ============================================================
# PARAGRAPH SPLITTING
# ============================================================

def split_paragraphs(text):
    """
    Split the document into logical paragraphs.
    """

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


# ============================================================
# OVERLAP CREATION
# ============================================================

def get_overlap_text(
    text,
    overlap_chars
):
    """
    Get the last complete words from a chunk.

    We never cut a word when creating overlap.
    """

    if len(text) <= overlap_chars:

        return text

    candidate = text[
        -overlap_chars:
    ]

    # Move to the beginning of the next complete word
    first_space = candidate.find(
        " "
    )

    if first_space != -1:

        candidate = candidate[
            first_space + 1:
        ]

    return candidate.strip()


def add_overlap(
    chunks
):
    """
    Add contextual overlap between neighboring chunks.

    The overlap is word-safe.
    """

    if not chunks:
        return []


    result = [
        chunks[0]
    ]


    for index in range(
        1,
        len(chunks)
    ):

        previous = chunks[
            index - 1
        ]

        current = chunks[
            index
        ]


        overlap = get_overlap_text(
            previous,
            OVERLAP_CHARS
        )


        if not overlap:

            result.append(
                current
            )

            continue


        combined = (
            overlap
            + "\n"
            + current
        )


        # Keep within maximum size
        if len(combined) > MAX_CHARS:

            # If overlap makes the chunk too large,
            # reduce it further.
            smaller_overlap = get_overlap_text(
                previous,
                120
            )

            combined = (
                smaller_overlap
                + "\n"
                + current
            )


        result.append(
            combined.strip()
        )


    return result


# ============================================================
# MERGE SMALL CHUNKS
# ============================================================

def merge_small_chunks(
    chunks
):
    """
    Merge very small chunks with neighboring chunks
    where possible.
    """

    if not chunks:
        return []


    final_chunks = []


    for chunk in chunks:

        chunk = chunk.strip()

        if not chunk:
            continue


        # ----------------------------------------------------
        # Small chunk
        # ----------------------------------------------------

        if (
            len(chunk) < MIN_CHARS
            and final_chunks
        ):

            proposed = (
                final_chunks[-1]
                + "\n"
                + chunk
            )


            if len(proposed) <= MAX_CHARS:

                final_chunks[-1] = (
                    proposed.strip()
                )

            else:

                final_chunks.append(
                    chunk
                )

        else:

            final_chunks.append(
                chunk
            )


    return final_chunks


# ============================================================
# CHUNK ONE DOCUMENT
# ============================================================

def chunk_document(
    document
):
    """
    Convert a complete policy document into chunks.
    """

    content = normalize_text(
        document.get(
            "content",
            ""
        )
    )


    if not content:

        return []


    # --------------------------------------------------------
    # Split into paragraphs
    # --------------------------------------------------------

    paragraphs = split_paragraphs(
        content
    )


    chunks = []


    # --------------------------------------------------------
    # Process every paragraph
    # --------------------------------------------------------

    for paragraph in paragraphs:

        paragraph_chunks = (
            build_chunks_from_paragraph(
                paragraph
            )
        )

        chunks.extend(
            paragraph_chunks
        )


    # --------------------------------------------------------
    # Add overlap
    # --------------------------------------------------------

    chunks = add_overlap(
        chunks
    )


    # --------------------------------------------------------
    # Merge small chunks
    # --------------------------------------------------------

    chunks = merge_small_chunks(
        chunks
    )


    return chunks


# ============================================================
# VALIDATE CHUNK
# ============================================================

def validate_chunk(
    chunk_text
):
    """
    Basic quality checks.
    """

    if not chunk_text:
        return False

    if not chunk_text.strip():
        return False

    # Make sure chunk does not start/end with obvious
    # whitespace artifacts.
    if chunk_text != chunk_text.strip():
        return False

    return True


# ============================================================
# CREATE CHUNK METADATA
# ============================================================

def create_chunk_record(
    document,
    chunk_text,
    chunk_index
):
    """
    Create the final metadata structure for a chunk.
    """

    return {

        "chunk_id":
            (
                f"{document['document_id']}"
                f"_chunk_"
                f"{chunk_index:04d}"
            ),

        "document_id":
            document[
                "document_id"
            ],

        "title":
            document[
                "title"
            ],

        "category":
            document[
                "category"
            ],

        "country":
            document[
                "country"
            ],

        "policy_type":
            document[
                "policy_type"
            ],

        "source":
            document[
                "source"
            ],

        "source_url":
            document[
                "source_url"
            ],

        "chunk_index":
            chunk_index,

        "chunk_text":
            chunk_text,

        "character_count":
            len(chunk_text),

        "word_count":
            len(
                chunk_text.split()
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "GITLAB HR POLICY CHUNKING"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print(
            "\nERROR: Input file not found:"
        )

        print(
            INPUT_FILE
        )

        return


    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(
            file
        )


    print(
        f"\nDocuments loaded: "
        f"{len(documents)}"
    )


    # --------------------------------------------------------
    # Generate chunks
    # --------------------------------------------------------

    all_chunks = []


    for doc_index, document in enumerate(
        documents,
        start=1
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"DOCUMENT "
            f"{doc_index}/{len(documents)}"
        )

        print(
            f"Title: "
            f"{document['title']}"
        )


        document_chunks = chunk_document(
            document
        )


        print(
            f"Chunks created: "
            f"{len(document_chunks)}"
        )


        # ----------------------------------------------------
        # Create metadata
        # ----------------------------------------------------

        valid_index = 1


        for chunk_text in document_chunks:

            if not validate_chunk(
                chunk_text
            ):

                continue


            chunk = create_chunk_record(
                document,
                chunk_text,
                valid_index
            )


            all_chunks.append(
                chunk
            )


            valid_index += 1


    # --------------------------------------------------------
    # Save chunks
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    total_characters = sum(
        chunk[
            "character_count"
        ]
        for chunk in all_chunks
    )


    total_words = sum(
        chunk[
            "word_count"
        ]
        for chunk in all_chunks
    )


    if all_chunks:

        average_size = (
            total_characters
            / len(all_chunks)
        )

    else:

        average_size = 0


    # --------------------------------------------------------
    # Category distribution
    # --------------------------------------------------------

    category_counts = {}


    for chunk in all_chunks:

        category = chunk[
            "category"
        ]

        category_counts[
            category
        ] = (
            category_counts.get(
                category,
                0
            )
            + 1
        )


    # --------------------------------------------------------
    # Country distribution
    # --------------------------------------------------------

    country_counts = {}


    for chunk in all_chunks:

        country = chunk[
            "country"
        ]

        country_counts[
            country
        ] = (
            country_counts.get(
                country,
                0
            )
            + 1
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CHUNKING COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        f"\nDocuments:"
        f" {len(documents)}"
    )


    print(
        f"Total chunks:"
        f" {len(all_chunks)}"
    )


    print(
        f"Total characters:"
        f" {total_characters:,}"
    )


    print(
        f"Total words:"
        f" {total_words:,}"
    )


    print(
        f"Average chunk size:"
        f" {average_size:.0f} characters"
    )


    print(
        f"\nSaved to:"
        f"\n{OUTPUT_FILE}"
    )


    # ========================================================
    # CHUNKS BY CATEGORY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CHUNKS BY CATEGORY"
    )

    print(
        "=" * 70
    )


    for category, count in sorted(
        category_counts.items()
    ):

        print(
            f"{category}: "
            f"{count}"
        )


    # ========================================================
    # CHUNKS BY COUNTRY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CHUNKS BY COUNTRY"
    )

    print(
        "=" * 70
    )


    for country, count in sorted(
        country_counts.items()
    ):

        print(
            f"{country}: "
            f"{count}"
        )


    # ========================================================
    # SAMPLE CHUNKS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SAMPLE CHUNKS"
    )

    print(
        "=" * 70
    )


    for chunk in all_chunks[:3]:

        print(
            "\n" + "-" * 70
        )


        print(
            f"Chunk ID: "
            f"{chunk['chunk_id']}"
        )


        print(
            f"Title: "
            f"{chunk['title']}"
        )


        print(
            f"Category: "
            f"{chunk['category']}"
        )


        print(
            f"Country: "
            f"{chunk['country']}"
        )


        print(
            f"Policy Type: "
            f"{chunk['policy_type']}"
        )


        print(
            f"Characters: "
            f"{chunk['character_count']}"
        )


        print(
            "\nText:"
        )


        print(
            chunk[
                "chunk_text"
            ][:1000]
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()