import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://handbook.gitlab.com"

START_URL = (
    "https://handbook.gitlab.com/"
    "handbook/people-policies/"
)

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "gitlab_people_policies.json"

# Maximum number of policy pages to collect
MAX_DOCUMENTS = 50

# Delay between requests
REQUEST_DELAY = 0.5

# Number of retries for temporary connection errors
MAX_RETRIES = 3


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    )
}


# ============================================================
# SESSION
# ============================================================

session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# TEXT CLEANING
# ============================================================

def fix_encoding(text):
    """
    Fix common UTF-8 decoding artifacts.
    """

    try:
        return text.encode("latin1").decode("utf-8")

    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def clean_text(text):
    """
    Clean extracted text.
    """

    text = fix_encoding(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

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

    return "\n".join(lines).strip()


# ============================================================
# DOWNLOAD PAGE
# ============================================================

def download_page(url):
    """
    Download webpage with retry handling.
    """

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = session.get(
                url,
                timeout=30
            )

            response.raise_for_status()

            return response.text

        except requests.RequestException as error:

            print(
                f"Request failed "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            print(error)

            if attempt < MAX_RETRIES:

                time.sleep(
                    attempt * 2
                )

            else:

                raise


# ============================================================
# FIND CONTENT CONTAINER
# ============================================================

def find_content_container(soup):
    """
    Find the actual article/content area.

    GitLab's handbook contains a very large navigation tree.
    We must avoid using the navigation as document data.
    """

    # Try common article/content containers
    selectors = [
        "article",
        "main article",
        ".content",
        ".markdown-body",
        ".handbook-content",
    ]

    for selector in selectors:

        element = soup.select_one(
            selector
        )

        if element:
            return element

    # Fallback: main
    main = soup.find("main")

    if main:
        return main

    return soup


# ============================================================
# EXTRACT TITLE
# ============================================================

def extract_title(soup, fallback_url):
    """
    Extract page title.
    """

    h1 = soup.find("h1")

    if h1:

        title = clean_text(
            h1.get_text(
                " ",
                strip=True
            )
        )

        if title:
            return title

    if soup.title:

        title = clean_text(
            soup.title.get_text(
                " ",
                strip=True
            )
        )

        if title:
            return title

    path = urlparse(
        fallback_url
    ).path

    name = (
        path.rstrip("/")
        .split("/")[-1]
    )

    return name.replace(
        "-",
        " "
    ).title()


# ============================================================
# NORMALIZE URL
# ============================================================

def normalize_url(url):
    """
    Normalize a URL by removing fragments and
    unnecessary trailing slashes.
    """

    url = url.split("#")[0]

    parsed = urlparse(url)

    normalized = (
        f"{parsed.scheme}://"
        f"{parsed.netloc}"
        f"{parsed.path}"
    )

    return normalized.rstrip("/")


# ============================================================
# VALIDATE HANDBOOK URL
# ============================================================

def is_handbook_url(url):
    """
    Keep only GitLab Handbook pages.
    """

    parsed = urlparse(url)

    if parsed.netloc != "handbook.gitlab.com":
        return False

    if not parsed.path.startswith("/handbook/"):
        return False

    return True


# ============================================================
# EXTRACT POLICY LINKS
# ============================================================

def discover_policy_links(soup):
    """
    Extract links from the actual People Policies content.

    We deliberately avoid the giant handbook navigation.
    """

    links = []

    content = find_content_container(
        soup
    )

    # --------------------------------------------------------
    # Find the main People Policies heading
    # --------------------------------------------------------

    h1 = content.find("h1")

    if not h1:

        # Try whole page
        h1 = soup.find("h1")

    if not h1:

        print(
            "WARNING: Could not locate H1."
        )

    # --------------------------------------------------------
    # Collect links
    # --------------------------------------------------------

    for anchor in content.find_all(
        "a",
        href=True
    ):

        href = anchor.get(
            "href",
            ""
        ).strip()

        link_text = clean_text(
            anchor.get_text(
                " ",
                strip=True
            )
        )

        if not href:
            continue

        if href.startswith(
            "#"
        ):
            continue

        absolute_url = urljoin(
            BASE_URL,
            href
        )

        absolute_url = normalize_url(
            absolute_url
        )

        if not is_handbook_url(
            absolute_url
        ):
            continue

        # Don't include the index itself
        if (
            absolute_url
            == START_URL.rstrip("/")
        ):
            continue

        # Skip obvious website utility pages
        blocked_terms = [
            "/about/",
            "/acquisitions/",
            "/alliances/",
            "/board-meetings/",
            "/business-technology/",
            "/customer-success/",
            "/engineering/",
            "/finance/",
            "/marketing/",
            "/sales/",
            "/security/",
            "/legal/",
        ]

        if any(
            term in absolute_url
            for term in blocked_terms
        ):
            continue

        links.append(
            {
                "url": absolute_url,
                "link_text": link_text
            }
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = {}

    for item in links:

        unique[
            item["url"]
        ] = item

    return list(
        unique.values()
    )


# ============================================================
# EXTRACT PAGE CONTENT
# ============================================================

def extract_page_content(
    html
):
    """
    Extract clean readable content.
    """

    soup = BeautifulSoup(
        html,
        "lxml"
    )

    # Remove unwanted elements
    for element in soup.find_all(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
        ]
    ):

        element.decompose()

    content = find_content_container(
        soup
    )

    text = content.get_text(
        "\n",
        strip=True
    )

    return (
        soup,
        clean_text(text)
    )


# ============================================================
# SCRAPE ONE DOCUMENT
# ============================================================

def scrape_document(
    url,
    link_text=""
):

    try:

        html = download_page(
            url
        )

        soup, content = (
            extract_page_content(
                html
            )
        )

        title = extract_title(
            soup,
            url
        )

        if not content:

            print(
                "WARNING: Empty document."
            )

            return None

        document = {
            "document_id": None,
            "title": title,
            "category": "People Policies",
            "link_text": link_text,
            "source_url": url,
            "content": content,
            "character_count": len(
                content
            ),
        }

        return document

    except Exception as error:

        print(
            f"ERROR scraping:\n{url}"
        )

        print(error)

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "GITLAB PEOPLE POLICY DATA COLLECTION"
    )
    print("=" * 70)

    print(
        f"\nStarting page:\n"
        f"{START_URL}"
    )

    # --------------------------------------------------------
    # Download index
    # --------------------------------------------------------

    try:

        index_html = download_page(
            START_URL
        )

    except Exception as error:

        print(
            "\nFAILED TO DOWNLOAD START PAGE"
        )

        print(error)

        return

    # --------------------------------------------------------
    # Parse index
    # --------------------------------------------------------

    index_soup = BeautifulSoup(
        index_html,
        "lxml"
    )

    # --------------------------------------------------------
    # Discover policy links
    # --------------------------------------------------------

    discovered = discover_policy_links(
        index_soup
    )

    print(
        f"\nCandidate policy links found: "
        f"{len(discovered)}"
    )

    # Show candidates before scraping
    print(
        "\nFirst candidate links:"
    )

    for index, item in enumerate(
        discovered[:20],
        start=1
    ):

        print(
            f"{index}. "
            f"{item['link_text']} "
            f"-> "
            f"{item['url']}"
        )

    # --------------------------------------------------------
    # Limit collection
    # --------------------------------------------------------

    selected = discovered[
        :MAX_DOCUMENTS
    ]

    print(
        f"\nWill scrape: "
        f"{len(selected)} documents"
    )

    # --------------------------------------------------------
    # Scrape documents
    # --------------------------------------------------------

    documents = []

    for index, item in enumerate(
        selected,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"DOCUMENT "
            f"{index}/{len(selected)}"
        )

        print(
            f"Link text: "
            f"{item['link_text']}"
        )

        print(
            f"URL: "
            f"{item['url']}"
        )

        document = scrape_document(
            item["url"],
            item["link_text"]
        )

        if document:

            document[
                "document_id"
            ] = (
                f"gitlab_policy_"
                f"{len(documents) + 1:04d}"
            )

            documents.append(
                document
            )

            print(
                f"Title: "
                f"{document['title']}"
            )

            print(
                f"Characters: "
                f"{document['character_count']:,}"
            )

        time.sleep(
            REQUEST_DELAY
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_characters = sum(
        document[
            "character_count"
        ]
        for document in documents
    )

    print("\n" + "=" * 70)
    print(
        "SCRAPING COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nDocuments collected: "
        f"{len(documents)}"
    )

    print(
        f"Total characters: "
        f"{total_characters:,}"
    )

    print(
        f"\nSaved to:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        "\nCollected documents:"
    )

    for index, document in enumerate(
        documents,
        start=1
    ):

        print(
            f"{index}. "
            f"{document['title']}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()