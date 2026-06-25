from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parent

BETTERBEE_GLOSSARIES_DIR = PROJECT_ROOT / "beekeeping_corpus" / "glossaries" / "betterbee_glossary"
GLOSSARIES_DIR = PROJECT_ROOT / "beekeeping_corpus" / "glossaries"
LAPPES_DIR = GLOSSARIES_DIR / "lappes_glossary"
GL1_FILE = GLOSSARIES_DIR / "gl1.txt"

BETTERBEE_GLOSSARIES_HTML = BETTERBEE_GLOSSARIES_DIR / "Glossary_of_Beekeeping_Terms_Betterbee.htm"
BETTERBEE_GLOSSARIES_JSON = BETTERBEE_GLOSSARIES_DIR / "betterbee_glossary_terms.json"
PREVIOUS_LAPPES_JSON = LAPPES_DIR / "lappes_glossary_terms.json"
OUTPUT_FILE = GLOSSARIES_DIR / "merged_glossary_terms.json"

counter = 1

LAPPES_BASE_URL = (
    "https://www.lappesbeesupply.com/beekeeping-blog/"
    "beekeeping-glossary-a-to-z-letter-{letter}"
)

SKIP_TERMS = {
    f"Beekeeping Glossary {letter}"
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
}

def re_clean_definition(full_text, concept_name):
    """Elegánsan eltávolítja a fogalom nevét és az írásjeleket a mondat elejéről."""
    # Ha a szöveg a fogalom nevével kezdődik
    if full_text.lower().startswith(concept_name.lower()):
        clean = full_text[len(concept_name):].strip()
        # Eltávolítjuk a kezdő kettőspontot, gondolatjelet vagy pontot
        clean = clean.lstrip('.:-–— ')
        return clean
    return full_text

def bootstrap_ontology_from_html(html_file_path, output_json_path=None):
    if not os.path.exists(html_file_path):
        print(f"Hiba: A '{html_file_path}' fájl nem található!")
        return

    with open(html_file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    extracted_ontology = []

    # 1. Megkeressük az összes <p> taget, ami egy-egy fogalmat reprezentál
    p_tags = soup.find_all('p')
    print(f"HTML elemzése: {len(p_tags)} db <p> tag találva.")

    for p in p_tags:
        # 2. Kivonjuk az első <strong> taget mint a fogalom nevét
        strong_tag = p.find('strong')

        # Ha nincs benne strong tag, vagy üres, átugorjuk (nem szótári bejegyzés)
        if not strong_tag:
            continue

        concept_name = strong_tag.get_text().strip()
        # Tisztítás: ha a végén kettőspont vagy pont van, levágjuk
        concept_name = concept_name.rstrip('.:')

        if len(concept_name) <= 1:
            continue

        # 3. Kigyűjtjük a belső extra tageket (pl. más fogalmakra való utalások a szövegben)
        # Minden olyan további strong, em vagy a tag, ami NEM az első definíciós szó
        cross_references = []
        for extra_tag in p.find_all(['strong', 'a', 'em']):
            tag_text = extra_tag.get_text().strip()
            # Ha nem a fő fogalomról van szó, és értelmes hosszúságú, elmentjük keresztutalásnak
            if tag_text.lower() != concept_name.lower() and len(tag_text) > 2:
                cross_references.append(tag_text)

        # Duplikációk kiszűrése a keresztutalásokból
        cross_references = list(set(cross_references))

        # 4. A definíció kinyerése: a teljes <p> szövegéből levágjuk a fogalom nevét
        full_p_text = p.get_text().strip()
        # Levágjuk az elejéről a fogalmat (esetleges kettősponttal együtt)
        definition = re_clean_definition(full_p_text, concept_name)

        # Ha a definíció túl rövid vagy hiányzik, adunk neki egy alap leírást
        if len(definition) < 5:
            definition = f"A specialized term in apiculture representing {concept_name}."

        # 5. Strukturált rekord összeállítása
        extracted_ontology.append(
            make_entry(
                term= concept_name.replace(u'\xa0', u' '),
                definition=definition.replace(u'\xa0', u' '),
                source="BettetBee",
                synonyms=[],
                cross_references= [ cf.replace(u'\xa0', u' ') for cf in cross_references ] # A belső tagekből kinyert kapcsolódó fogalmak
            )
        )

        #counter += 1

    # Elmentjük az ontológia alapfájlt
    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_ontology, f, indent=2, ensure_ascii=False)


    print(f"Sikeresen inicializálva {len(extracted_ontology)} fogalom.")
    print(f"Mentve a(z) '{output_json_path}' fájlba.")
    return extracted_ontology



def clean_text(text: str) -> str:
    """Normalize whitespace, non-breaking spaces, and soft hyphen artifacts."""
    text = text.replace("\xa0", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_term_key(term: str) -> str:
    """
    Normalize terms for duplicate detection.

    This keeps display terms intact in output, but compares duplicates more safely.
    """
    term = clean_text(term)
    term = term.strip(" .:;,-–—")
    term = re.sub(r"\s+", " ", term)
    return term.casefold()


def source_label(path: Path | str) -> str:
    return Path(path).name


def make_entry(term: str, definition: str, source: str,synonyms:list[str]=[],cross_references:list[str]=[] ) -> dict[str, Any]:
    return {
        "name": clean_text(term).strip(" .:;,-–—"),
        "definition": clean_text(definition),
        "sources": [source],
        "synonyms":synonyms,
        "cross_references":cross_references
    }


def merge_entry(
    merged: dict[str, dict[str, Any]],
    entry: dict[str, Any],
) -> None:
    """
    Merge a glossary entry by normalized term.

    If a term appears in multiple sources, keep the longest definition.
    Preserve all source names in `sources`.
    """
    term = clean_text(entry.get("name", ""))
    definition = clean_text(entry.get("definition", ""))

    if not term or not definition:
        return

    key = normalize_term_key(term)

    incoming_sources = entry.get("sources") or []
    if not incoming_sources and entry.get("source_file"):
        incoming_sources = [entry["source_file"]]
    if not incoming_sources and entry.get("source"):
        incoming_sources = [entry["source"]]

    if key not in merged:
        merged[key] = {
            "name": term,
            "definition": definition,
            "sources": sorted(set(incoming_sources)),
        }
        return

    existing = merged[key]

    existing_sources = set(existing.get("sources", []))
    existing_sources.update(incoming_sources)
    existing["sources"] = sorted(existing_sources)

    if len(definition) > len(existing.get("definition", "")):
        existing["definition"] = definition
        # Prefer the term spelling associated with the longest definition.
        existing["term"] = term


def get_term_from_h2(h2: Tag) -> str:
    """
    Extract only visible text from an h2.

    Images inside headings are ignored automatically by get_text().
    """
    return clean_text(h2.get_text(" ", strip=True))


def is_probable_glossary_term(term: str) -> bool:
    """
    Filter out non-entry headings such as page titles, section comments,
    empty image-only headings, and editorial headings.
    """
    if not term:
        return False

    if term in SKIP_TERMS:
        return False

    lowered = term.lower()

    skip_fragments = (
        "read on",
        "keep reading",
        "in conclusion",
        "related articles",
        "letter ",
        "a to z",
        "next →",
        "previous",
    )

    if any(fragment in lowered for fragment in skip_fragments):
        return False

    # Real glossary terms are usually short. This also filters accidental headings.
    if len(term.split()) > 8:
        return False

    return True


def extract_definition_after_h2(h2: Tag) -> str:
    """
    Collect definition paragraphs after a glossary term heading.

    Stops when the next h2 appears. This handles definitions split across
    multiple <p> tags.
    """
    paragraphs: list[str] = []

    for sibling in h2.find_next_siblings():
        if not isinstance(sibling, Tag):
            continue

        if sibling.name == "h2":
            break

        if sibling.name != "p":
            continue

        paragraph_text = clean_text(sibling.get_text(" ", strip=True))

        if not paragraph_text:
            continue

        lowered = paragraph_text.lower()

        # Skip editorial connector text that is not part of a definition.
        if lowered.startswith(
            (
                "read on",
                "keep reading",
                "if you're finding",
                "in conclusion",
            )
        ):
            continue

        paragraphs.append(paragraph_text)

    return clean_text(" ".join(paragraphs))


def extract_glossary_entries_from_html_file(file_path: Path) -> list[dict[str, Any]]:
    html = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Prefer the actual blog/glossary body when present.
    content = (
        soup.select_one(".dib-post-content")
        or soup.select_one("#dib-post-single")
        or soup.body
        or soup
    )

    entries: list[dict[str, Any]] = []

    for h2 in content.find_all("h2"):
        term = get_term_from_h2(h2)

        if not is_probable_glossary_term(term):
            continue

        definition = extract_definition_after_h2(h2)

        if not definition:
            continue

        entries.append(
            make_entry(
                term=term,
                definition=definition,
                source=source_label(file_path),
            )
        )

    return entries


def parse_lappes_html_directory(input_dir: Path = LAPPES_DIR) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    html_files = sorted(input_dir.glob("*.htm"))

    for file_path in html_files:
        file_entries = extract_glossary_entries_from_html_file(file_path)
        entries.extend(file_entries)
        print(f"{file_path.name}: extracted {len(file_entries)} HTML entries")

    return entries


def parse_previous_json(json_path: Path = PREVIOUS_LAPPES_JSON) -> list[dict[str, Any]]:
    """
    Load entries produced by a previous extraction step.

    Supports both:
    - {"term": "...", "definition": "..."}
    - {"name": "...", "definition": "..."}
    """
    if not json_path.exists():
        print(f"Previous JSON not found, skipping: {json_path}")
        return []

    raw = json.loads(json_path.read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in {json_path}, got {type(raw).__name__}")

    entries: list[dict[str, Any]] = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        term = item.get("term") or item.get("name")
        definition = item.get("definition")

        if not term or not definition:
            continue

        entries.append(
            make_entry(
                term=str(term),
                definition=str(definition),
                source=source_label(json_path),
            )
        )

    print(f"{json_path.name}: loaded {len(entries)} previous JSON entries")
    return entries


def split_gl1_entry(line: str) -> tuple[str, str] | None:
    """
    Split a gl1.txt entry into term and definition.

    The source uses an em dash:
        Term—definition

    This also accepts en dash or hyphen with surrounding spaces as a fallback.
    """
    line = clean_text(line)

    if not line or line.startswith("#"):
        return None

    if "—" in line:
        term, definition = line.split("—", 1)
    elif "–" in line:
        term, definition = line.split("–", 1)
    else:
        # Conservative fallback: only split hyphen when surrounded by spaces.
        match = re.match(r"^(.+?)\s+-\s+(.+)$", line)
        if not match:
            return None
        term, definition = match.groups()

    term = clean_text(term).strip(" .:;,-–—")
    definition = clean_text(definition).strip(" .:;,-–—")

    if not term or not definition:
        return None

    return term, definition


def parse_gl1_txt(gl1_path: Path = GL1_FILE) -> list[dict[str, Any]]:
    """
    Parse gl1.txt, where each entry starts with:
        Term—definition

    Lines without a dash are treated as continuation lines for the previous term.
    """
    if not gl1_path.exists():
        print(f"gl1.txt not found, skipping: {gl1_path}")
        return []

    entries: list[dict[str, Any]] = []

    current_term: str | None = None
    current_definition_parts: list[str] = []

    def flush_current_entry() -> None:
        nonlocal current_term, current_definition_parts

        if current_term and current_definition_parts:
            entries.append(
                make_entry(
                    term=current_term,
                    definition=" ".join(current_definition_parts),
                    source=source_label(gl1_path),
                )
            )

        current_term = None
        current_definition_parts = []

    for raw_line in gl1_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = clean_text(raw_line)

        if not line or line.startswith("#"):
            continue

        parsed = split_gl1_entry(line)

        if parsed:
            flush_current_entry()
            current_term, definition = parsed
            current_definition_parts = [definition]
        elif current_term:
            # Continuation line for the previous definition.
            current_definition_parts.append(line)

    flush_current_entry()

    print(f"{gl1_path.name}: extracted {len(entries)} text entries")
    return entries


def merge_glossary_sources(
    include_html: bool = True,
    include_previous_json: bool = False,
    include_gl1: bool = True,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    source_entries: list[dict[str, Any]] = []
    source_entries.extend(bootstrap_ontology_from_html(BETTERBEE_GLOSSARIES_HTML))

    if include_html:
        source_entries.extend(parse_lappes_html_directory())

    if include_previous_json:
        source_entries.extend(parse_previous_json())

    if include_gl1:
        source_entries.extend(parse_gl1_txt())

    for entry in source_entries:
        merge_entry(merged, entry)

    final_entries = sorted(
        merged.values(),
        key=lambda item: normalize_term_key(item["name"]),
    )
    for i,entry in enumerate(final_entries):
        entry["id"] = f"ONT_{str(i).zfill(3)}"

    return final_entries


def write_merged_glossary(output_file: Path = OUTPUT_FILE) -> None:
    final_entries = merge_glossary_sources()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(final_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nExtracted {len(final_entries)} unique glossary terms")
    print(f"Wrote: {output_file}")


def scrape_dynamic_glossary_files(
    output_dir: Path = LAPPES_DIR,
    start_letter: str = "A",
    end_letter: str = "Z",
) -> None:
    """
    Optional helper to render and save dynamic glossary pages.

    This keeps browser/page lifetime safe:
    - create browser once
    - create one page per URL
    - close page after saving
    - close browser only after all pages are done
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Launching browser...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        try:
            for codepoint in range(ord(start_letter.upper()), ord(end_letter.upper()) + 1):
                letter = chr(codepoint)
                url = LAPPES_BASE_URL.format(letter=letter.lower())
                output_path = output_dir / f"lappes_glossary_beekeeping-glossary-a-to-z-letter-{letter}.htm"

                print(f"Navigating to: {url}")

                page = browser.new_page()

                try:
                    page.goto(url, wait_until="networkidle", timeout=60_000)
                    output_path.write_text(page.content(), encoding="utf-8")
                    print(f"Saved: {output_path}")
                finally:
                    page.close()
        finally:
            browser.close()


if __name__ == "__main__":
    write_merged_glossary()