import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_FILE = Path(
    "data/vector_store/policy_faiss.index"
)

METADATA_FILE = Path(
    "data/vector_store/policy_metadata.json"
)

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


# ============================================================
# RETRIEVER
# ============================================================

class PolicyRetriever:

    def __init__(
        self,
        index_file=INDEX_FILE,
        metadata_file=METADATA_FILE,
        model_name=MODEL_NAME
    ):

        print(
            "\nLoading FAISS index..."
        )

        self.index = faiss.read_index(
            str(index_file)
        )


        print(
            "Loading metadata..."
        )

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(
                file
            )


        print(
            "Loading embedding model..."
        )

        self.model = SentenceTransformer(
            model_name
        )


        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if (
            self.index.ntotal
            != len(self.metadata)
        ):

            raise ValueError(
                "FAISS index and metadata "
                "have different numbers of records."
            )


        print(
            f"\nRetriever ready."
        )

        print(
            f"Vectors: {self.index.ntotal}"
        )


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=5,
        country=None,
        category=None
    ):

        if not query.strip():

            return []


        # ----------------------------------------------------
        # Convert question to embedding
        # ----------------------------------------------------

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )


        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )


        # ----------------------------------------------------
        # Search more candidates if metadata filtering
        # is requested.
        # ----------------------------------------------------

        if country or category:

            search_k = min(
                top_k * 5,
                self.index.ntotal
            )

        else:

            search_k = min(
                top_k,
                self.index.ntotal
            )


        scores, indices = self.index.search(
            query_embedding,
            search_k
        )


        results = []


        # ----------------------------------------------------
        # Process results
        # ----------------------------------------------------

        for score, index_position in zip(
            scores[0],
            indices[0]
        ):

            if index_position < 0:
                continue


            metadata = self.metadata[
                index_position
            ]


            # ------------------------------------------------
            # Metadata filters
            # ------------------------------------------------

            if country:

                if (
                    metadata.get(
                        "country"
                    )
                    != country
                ):

                    continue


            if category:

                if (
                    metadata.get(
                        "category"
                    )
                    != category
                ):

                    continue


            result = {

                "rank":
                    len(results) + 1,

                "score":
                    float(score),

                "chunk_id":
                    metadata.get(
                        "chunk_id"
                    ),

                "document_id":
                    metadata.get(
                        "document_id"
                    ),

                "title":
                    metadata.get(
                        "title"
                    ),

                "category":
                    metadata.get(
                        "category"
                    ),

                "country":
                    metadata.get(
                        "country"
                    ),

                "policy_type":
                    metadata.get(
                        "policy_type"
                    ),

                "source":
                    metadata.get(
                        "source"
                    ),

                "source_url":
                    metadata.get(
                        "source_url"
                    ),

                "chunk_index":
                    metadata.get(
                        "chunk_index"
                    ),

                "text":
                    metadata.get(
                        "chunk_text"
                    )

            }


            results.append(
                result
            )


            if len(results) >= top_k:

                break


        return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    query,
    results
):

    print(
        "\n" + "=" * 80
    )

    print(
        "SEARCH QUERY"
    )

    print(
        "=" * 80
    )

    print(
        f"\n{query}"
    )


    if not results:

        print(
            "\nNo results found."
        )

        return


    print(
        "\n" + "=" * 80
    )

    print(
        "TOP RESULTS"
    )

    print(
        "=" * 80
    )


    for result in results:

        print(
            "\n" + "-" * 80
        )

        print(
            f"Rank: "
            f"{result['rank']}"
        )

        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Title: "
            f"{result['title']}"
        )

        print(
            f"Category: "
            f"{result['category']}"
        )

        print(
            f"Country: "
            f"{result['country']}"
        )

        print(
            f"Policy Type: "
            f"{result['policy_type']}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            "\nTEXT:"
        )

        print(
            result["text"]
        )


# ============================================================
# TEST QUESTIONS
# ============================================================

def run_tests():

    retriever = PolicyRetriever()


    test_questions = [

        "What is GitLab's policy regarding hiring family members?",

        "What should an employee do if they experience harassment?",

        "What is the policy for taking time off?",

        "What happens when an employee leaves GitLab?",

        "What are GitLab's remote work rules in France?",

    ]


    for question in test_questions:

        results = retriever.search(
            question,
            top_k=5
        )

        print_results(
            question,
            results
        )


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode():

    retriever = PolicyRetriever()


    print(
        "\n" + "=" * 80
    )

    print(
        "GITLAB HR POLICY SEMANTIC SEARCH"
    )

    print(
        "=" * 80
    )

    print(
        "\nType your HR policy question."
    )

    print(
        "Type 'exit' to quit."
    )


    while True:

        print(
            "\n" + "-" * 80
        )

        query = input(
            "\nQuestion: "
        ).strip()


        if query.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "\nGoodbye."
            )

            break


        if not query:

            continue


        results = retriever.search(
            query,
            top_k=5
        )


        print_results(
            query,
            results
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nChoose mode:"
    )

    print(
        "1 = Run test questions"
    )

    print(
        "2 = Interactive search"
    )


    choice = input(
        "\nEnter 1 or 2: "
    ).strip()


    if choice == "1":

        run_tests()

    else:

        interactive_mode()