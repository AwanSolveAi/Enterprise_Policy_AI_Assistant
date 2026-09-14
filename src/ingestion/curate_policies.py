import json
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/raw/gitlab_people_policies.json"
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
    "hr_policies.json"
)


# ============================================================
# DOCUMENT SELECTION
# ============================================================

KEEP_KEYWORDS = [

    "people policies",
    "anti-harassment",
    "offboarding",
    "time off",
    "leave of absence",
    "right to disconnect",
    "remote work charter",
    "sexual harassment",
    "contracts",
    "probation",
    "employment",
    "benefits",
]


EXCLUDE_KEYWORDS = [

    "gitlab values",
    "personal vpn",
    "legal & corporate affairs",
    "handbook",
    "home office",
    "productive home office",
    "complete guide to starting a remote job",
    "leading through adversity",
    "modern health",
    "gitlab events code of conduct",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize extracted policy text.
    """

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(
        lines
    ).strip()


# ============================================================
# DOCUMENT SELECTION
# ============================================================

def should_keep(document):

    title = document.get(
        "title",
        ""
    ).lower()

    # Explicit exclusions
    for keyword in EXCLUDE_KEYWORDS:

        if keyword in title:
            return False

    # Useful HR documents
    for keyword in KEEP_KEYWORDS:

        if keyword in title:
            return True

    return False


# ============================================================
# CATEGORY CLASSIFICATION
# ============================================================

def classify_category(
    title,
    content
):

    title_lower = title.lower()

    # --------------------------------------------------------
    # OFFBOARDING
    # --------------------------------------------------------

    if any(
        keyword in title_lower
        for keyword in [
            "offboarding",
            "return of property",
        ]
    ):

        return "Offboarding"


    # --------------------------------------------------------
    # REMOTE WORK
    # --------------------------------------------------------

    if any(
        keyword in title_lower
        for keyword in [
            "remote work",
            "right to disconnect",
        ]
    ):

        return "Remote Work"


    # --------------------------------------------------------
    # WORKPLACE CONDUCT
    # --------------------------------------------------------

    if any(
        keyword in title_lower
        for keyword in [
            "anti-harassment",
            "sexual harassment",
        ]
    ):

        return "Workplace Conduct"


    # --------------------------------------------------------
    # LEAVE & TIME OFF
    # --------------------------------------------------------

    if any(
        keyword in title_lower
        for keyword in [
            "time off",
            "leave of absence",
        ]
    ):

        return "Leave & Time Off"


    # --------------------------------------------------------
    # BENEFITS
    # --------------------------------------------------------

    if "benefit" in title_lower:

        return "Benefits"


    # --------------------------------------------------------
    # EMPLOYMENT
    # --------------------------------------------------------

    if any(
        keyword in title_lower
        for keyword in [
            "contracts",
            "probation",
            "employment",
        ]
    ):

        return "Employment"


    # --------------------------------------------------------
    # PEOPLE POLICIES
    # --------------------------------------------------------

    if "people policies" in title_lower:

        return "People Policies"


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return "General HR"


# ============================================================
# COUNTRY DETECTION
# ============================================================

def detect_country(title):

    title_lower = title.lower()

    if "usa" in title_lower:
        return "United States"

    if "france" in title_lower:
        return "France"

    if "ireland" in title_lower:
        return "Ireland"

    if "india" in title_lower:
        return "India"

    if "australia" in title_lower:
        return "Australia"

    return "Global"


# ============================================================
# POLICY TYPE
# ============================================================

def detect_policy_type(
    title,
    content
):

    title_lower = title.lower()

    # Charters
    if "charter" in title_lower:
        return "Charter"

    # Policies
    if "policy" in title_lower:
        return "Policy"

    # Benefits
    if "benefit" in title_lower:
        return "Benefits Information"

    # Time off
    if "time off" in title_lower:
        return "Procedure"

    # Leave
    if "leave" in title_lower:
        return "Policy"

    # Offboarding
    if "offboarding" in title_lower:
        return "Procedure"

    # Contracts
    if "contract" in title_lower:
        return "Employment Policy"

    return "General Policy"


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_document(
    document,
    index
):

    title = document.get(
        "title",
        ""
    ).strip()

    content = document.get(
        "content",
        ""
    )

    content = normalize_text(
        content
    )

    category = classify_category(
        title,
        content
    )

    country = detect_country(
        title
    )

    policy_type = detect_policy_type(
        title,
        content
    )

    processed = {

        "document_id":
            f"hr_policy_{index:04d}",

        "title":
            title,

        "category":
            category,

        "country":
            country,

        "policy_type":
            policy_type,

        "source":
            "GitLab Public Handbook",

        "source_url":
            document.get(
                "source_url",
                ""
            ),

        "content":
            content,

        "character_count":
            len(content),

        "word_count":
            len(
                content.split()
            ),

    }

    return processed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "HR POLICY DATA CURATION"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print(
            "\nERROR: Input file not found:"
        )

        print(
            INPUT_FILE
        )

        return


    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        raw_documents = json.load(
            file
        )


    print(
        f"\nRaw documents: "
        f"{len(raw_documents)}"
    )


    # --------------------------------------------------------
    # Select documents
    # --------------------------------------------------------

    selected_documents = []

    excluded_documents = []


    for document in raw_documents:

        if should_keep(document):

            selected_documents.append(
                document
            )

        else:

            excluded_documents.append(
                document
            )


    print(
        f"Selected documents: "
        f"{len(selected_documents)}"
    )

    print(
        f"Excluded documents: "
        f"{len(excluded_documents)}"
    )


    # --------------------------------------------------------
    # Process documents
    # --------------------------------------------------------

    processed_documents = []


    for index, document in enumerate(
        selected_documents,
        start=1
    ):

        processed = process_document(
            document,
            index
        )

        processed_documents.append(
            processed
        )


    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            processed_documents,
            file,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------------------------------
    # Display selected documents
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTED HR DOCUMENTS"
    )

    print(
        "=" * 70
    )


    for index, document in enumerate(
        processed_documents,
        start=1
    ):

        print(
            f"{index}. "
            f"{document['title']}"
        )

        print(
            f"   Category: "
            f"{document['category']}"
        )

        print(
            f"   Country: "
            f"{document['country']}"
        )

        print(
            f"   Type: "
            f"{document['policy_type']}"
        )

        print(
            f"   Characters: "
            f"{document['character_count']:,}"
        )


    # --------------------------------------------------------
    # Display excluded documents
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EXCLUDED DOCUMENTS"
    )

    print(
        "=" * 70
    )


    for document in excluded_documents:

        print(
            f"- "
            f"{document.get('title', '')}"
        )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_characters = sum(
        document[
            "character_count"
        ]
        for document
        in processed_documents
    )

    total_words = sum(
        document[
            "word_count"
        ]
        for document
        in processed_documents
    )


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CURATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDocuments: "
        f"{len(processed_documents)}"
    )

    print(
        f"Total characters: "
        f"{total_characters:,}"
    )

    print(
        f"Total words: "
        f"{total_words:,}"
    )

    print(
        f"\nSaved to:"
        f"\n{OUTPUT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
    