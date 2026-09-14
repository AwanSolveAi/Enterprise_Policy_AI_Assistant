"""
BM25 Retriever for Enterprise Policy AI

Performs keyword-based retrieval over the HR policy chunks.

Pipeline:

Question
   ↓
Tokenization
   ↓
BM25
   ↓
Top Keyword Matches
"""

import os
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

CHUNKS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "policy_chunks.json"
)


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize(text):
    """
    Simple and reliable tokenizer for BM25.

    Converts text to lowercase and extracts words.
    """

    if not text:
        return []

    return re.findall(
        r"\b[a-zA-Z0-9']+\b",
        str(text).lower()
    )


# =============================================================================
# METADATA NORMALIZATION
# =============================================================================

def normalize_data(data):
    """
    Handles different possible JSON structures.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in [
            "chunks",
            "documents",
            "data",
            "items"
        ]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(
        "Unsupported policy_chunks.json structure."
    )


# =============================================================================
# FIELD HELPER
# =============================================================================

def get_field(item, *names, default=""):

    for name in names:

        if name in item:

            value = item[name]

            if value is not None:
                return str(value)

    return default


# =============================================================================
# BM25 RETRIEVER
# =============================================================================

class BM25Retriever:

    def __init__(self):

        print()
        print("=" * 80)
        print("INITIALIZING BM25 RETRIEVER")
        print("=" * 80)

        if not os.path.exists(CHUNKS_PATH):

            raise FileNotFoundError(
                f"Policy chunks not found:\n{CHUNKS_PATH}"
            )

        print()
        print("Loading policy chunks...")

        with open(
            CHUNKS_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            raw_data = json.load(f)

        self.chunks = normalize_data(raw_data)

        print(
            f"Chunks loaded: {len(self.chunks)}"
        )

        # -------------------------------------------------------------
        # PREPARE BM25 CORPUS
        # -------------------------------------------------------------

        print()
        print("Preparing BM25 corpus...")

        self.tokenized_corpus = []

        for chunk in self.chunks:

            text = get_field(
                chunk,
                "text",
                "content",
                "chunk_text"
            )

            tokens = tokenize(text)

            self.tokenized_corpus.append(tokens)

        # -------------------------------------------------------------
        # BUILD BM25 INDEX
        # -------------------------------------------------------------

        print("Building BM25 index...")

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print()
        print("BM25 retriever ready.")
        print(
            f"Indexed documents: {len(self.chunks)}"
        )

    # =========================================================================
    # SEARCH
    # =========================================================================

    def search(self, query, top_k=10):

        """
        Search policy chunks using BM25.

        Returns ranked results.
        """

        if not query.strip():
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # -------------------------------------------------------------
        # GET BM25 SCORES
        # -------------------------------------------------------------

        scores = self.bm25.get_scores(
            query_tokens
        )

        # -------------------------------------------------------------
        # SORT DESCENDING
        # -------------------------------------------------------------

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for idx in ranked_indices[:top_k]:

            score = float(scores[idx])

            # Ignore zero-score results
            if score <= 0:
                continue

            results.append({

                "rank": len(results) + 1,

                "index": int(idx),

                "score": score,

                "chunk": self.chunks[idx]

            })

        return results


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

def display_results(results):

    print()
    print("=" * 80)
    print("BM25 RESULTS")
    print("=" * 80)

    if not results:

        print()
        print("No results found.")
        return

    for result in results:

        chunk = result["chunk"]

        title = get_field(
            chunk,
            "title",
            "document_title",
            default="Unknown"
        )

        chunk_id = get_field(
            chunk,
            "chunk_id",
            "id",
            default="Unknown"
        )

        category = get_field(
            chunk,
            "category",
            default="Unknown"
        )

        text = get_field(
            chunk,
            "text",
            "content",
            "chunk_text"
        )

        print()
        print("-" * 80)
        print(
            f"Rank: {result['rank']}"
        )
        print(
            f"BM25 Score: {result['score']:.4f}"
        )
        print(
            f"Index: {result['index']}"
        )
        print(
            f"Chunk ID: {chunk_id}"
        )
        print(
            f"Title: {title}"
        )
        print(
            f"Category: {category}"
        )

        print()
        print("TEXT:")
        print(text[:1000])


# =============================================================================
# INTERACTIVE TEST
# =============================================================================

def interactive_mode():

    retriever = BM25Retriever()

    print()
    print("=" * 80)
    print("GITLAB HR POLICY BM25 SEARCH")
    print("=" * 80)

    print()
    print("Type your HR policy question.")
    print("Type 'exit' to quit.")

    while True:

        print()

        question = input(
            "Question: "
        ).strip()

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print()
            print("Goodbye.")
            break

        if not question:
            continue

        results = retriever.search(
            question,
            top_k=5
        )

        display_results(results)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    interactive_mode()