"""Scrapes SHL product catalogue pages, handles pagination, retries, and saves to JSON."""

import json
import time
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from config import settings

# Data Classes

@dataclass
class Assessment:
    """Represents a single SHL assessment (Individual Test Solution)."""
    name: str
    url: str
    description: str = ""
    job_levels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    assessment_length: str = ""
    test_type: list[str] = field(default_factory=list)
    remote_testing: bool = False
    adaptive_irt: bool = False

# HTTP helpers

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)
    return session

@retry(
    stop=stop_after_attempt(settings.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _get(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url, timeout=settings.REQUEST_TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")

# Catalogue page parser

BASE = "https://www.shl.com"

def _parse_flag(cell_soup: BeautifulSoup) -> bool:
    """Return True if a table cell contains a green check / tick icon."""
    # SHL uses aria-label="Yes" or an img with specific alt text
    yes = cell_soup.find(attrs={"aria-label": "Yes"})
    if yes:
        return True
    svg = cell_soup.find("svg")
    if svg and "check" in str(svg).lower():
        return True
    img = cell_soup.find("img")
    if img and img.get("alt", "").lower() in ("yes", "true", "✓", "check"):
        return True
    # SHL also uses a simple text "●" or a span with class
    span = cell_soup.find("span")
    if span:
        cls = " ".join(span.get("class", []))
        if "check" in cls or "yes" in cls or "active" in cls:
            return True
    return False

def _parse_catalogue_row(row: BeautifulSoup) -> Optional[dict]:
    """Parse one <tr> from the catalogue table into a dict."""
    cells = row.find_all("td")
    if len(cells) < 4:
        return None

    # Column 0: Assessment name + link
    name_cell = cells[0]
    a_tag = name_cell.find("a")
    if not a_tag:
        return None

    name = a_tag.get_text(strip=True)
    href = a_tag.get("href", "")
    if not href:
        return None
    url = href if href.startswith("http") else BASE + href

    # Column 1: Remote Testing support flag
    remote_testing = _parse_flag(cells[1])

    # Column 2: Adaptive / IRT flag
    adaptive_irt = _parse_flag(cells[2])

    # Column 3: Test-type badges (letters in <span> tags)
    test_type_cell = cells[3]
    test_types = [
        span.get_text(strip=True)
        for span in test_type_cell.find_all("span")
        if span.get_text(strip=True)
    ]

    return {
        "name": name,
        "url": url,
        "remote_testing": remote_testing,
        "adaptive_irt": adaptive_irt,
        "test_type": test_types,
    }

def _scrape_catalogue_page(
    session: requests.Session, start: int
) -> tuple[list[dict], bool]:
    """
    Fetch one paginated page of the Individual Test Solutions catalogue.
    Returns (list_of_row_dicts, has_more_pages).
    """
    url = (
        f"{settings.SHL_CATALOGUE_URL}"
        f"?start={start}&type=1"
    )
    logger.info(f"Fetching catalogue page  start={start}  URL={url}")
    soup = _get(session, url)

    rows: list[dict] = []

    # The catalogue renders inside a <table> with class "custom__table-striped"
    # or similar; we look for all <tr> that contain assessment data.
    table = soup.find("table")
    if not table:
        # Try alternate structure: individual cards
        logger.warning("No <table> found – attempting card-based parsing")
        return _parse_cards(soup), False

    tbody = table.find("tbody") or table
    for row in tbody.find_all("tr"):
        parsed = _parse_catalogue_row(row)
        if parsed:
            rows.append(parsed)

    # Detect pagination "next" button / link
    next_btn = soup.find("a", attrs={"aria-label": "Next"})
    if not next_btn:
        # Try generic pagination link
        next_btn = soup.find("li", class_=re.compile(r"next"))
        if next_btn:
            next_btn = next_btn.find("a")
    has_more = next_btn is not None and len(rows) > 0

    return rows, has_more

def _parse_cards(soup: BeautifulSoup) -> list[dict]:
    """Fallback: parse card-style catalogue layout."""
    cards = []
    for card in soup.find_all("div", class_=re.compile(r"product")):
        a = card.find("a")
        if not a:
            continue
        name = a.get_text(strip=True)
        href = a.get("href", "")
        if not href:
            continue
        url = href if href.startswith("http") else BASE + href
        cards.append({
            "name": name,
            "url": url,
            "remote_testing": False,
            "adaptive_irt": False,
            "test_type": [],
        })
    return cards

# Detail page parser

# Map of full test-type names (used on detail pages) to short codes
TEST_TYPE_MAP: dict[str, str] = {
    "ability & aptitude": "A",
    "biodata & situational judgment": "B",
    "competencies": "C",
    "development & 360": "D",
    "assessment exercises": "E",
    "knowledge & skills": "K",
    "personality & behavior": "P",
    "personality & behaviour": "P",
    "simulations": "S",
}

def _scrape_detail_page(
    session: requests.Session, assessment: dict
) -> dict:
    """Enrich an assessment dict with data from its individual detail page."""
    try:
        soup = _get(session, assessment["url"])
    except Exception as exc:
        logger.warning(f"Could not fetch detail page {assessment['url']}: {exc}")
        return assessment

    # Description – first substantial <p> in the main content area
    description = ""
    content_div = (
        soup.find("div", class_=re.compile(r"product-detail|content|description", re.I))
        or soup.find("main")
        or soup.find("article")
    )
    if content_div:
        for p in content_div.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50:
                description = text
                break
    assessment["description"] = description

    # Structured fields inside a definition list or key-value pairs
    def _find_field(label: str) -> str:
        """Find value next to a label anywhere in the page."""
        # Pattern 1: <dt> label → <dd> value
        for dt in soup.find_all("dt"):
            if label.lower() in dt.get_text(strip=True).lower():
                dd = dt.find_next_sibling("dd")
                if dd:
                    return dd.get_text(strip=True)
        # Pattern 2: <th> label in a table
        for th in soup.find_all("th"):
            if label.lower() in th.get_text(strip=True).lower():
                td = th.find_next_sibling("td")
                if td:
                    return td.get_text(strip=True)
        # Pattern 3: plain span / div pairs
        for elem in soup.find_all(["span", "div", "li"]):
            text = elem.get_text(strip=True)
            if text.lower().startswith(label.lower()):
                return text[len(label):].strip(":").strip()
        return ""

    # Assessment length / duration
    for label in ("Assessment length", "Duration", "Approximate Completion Time"):
        val = _find_field(label)
        if val:
            assessment["assessment_length"] = val
            break

    # Job levels
    job_levels_text = _find_field("Job level")
    if job_levels_text:
        assessment["job_levels"] = [j.strip() for j in re.split(r"[,;|]", job_levels_text) if j.strip()]

    # Languages
    lang_text = _find_field("Language")
    if lang_text:
        assessment["languages"] = [l.strip() for l in re.split(r"[,;|]", lang_text) if l.strip()]

    # Test type – override/supplement list from catalogue row
    if not assessment.get("test_type"):
        tt_text = _find_field("Test type")
        if not tt_text:
            tt_text = _find_field("Measures")
        if tt_text:
            code = TEST_TYPE_MAP.get(tt_text.lower(), tt_text[:1].upper())
            assessment["test_type"] = [code]

    time.sleep(settings.REQUEST_DELAY)
    return assessment

# Main scraper orchestrator

class SHLScraper:
    """Orchestrates full catalogue scraping and detail-page enrichment."""

    def __init__(self, output_path: Path = settings.RAW_DATA_PATH):
        self.output_path = Path(output_path)
        self.session = _make_session()

    # ------------------------------------------------------------------ #

    def scrape_catalogue_listings(self) -> list[dict]:
        """
        Paginate through the SHL catalogue and collect all
        Individual Test Solution rows.
        """
        all_rows: list[dict] = []
        start = 0
        page_size = 12  # SHL default items per page

        while True:
            rows, has_more = _scrape_catalogue_page(self.session, start)
            if not rows:
                logger.info(f"No rows returned at start={start}, stopping pagination.")
                break
            all_rows.extend(rows)
            logger.info(f"Collected {len(all_rows)} listings so far …")

            if not has_more:
                break

            start += page_size
            time.sleep(settings.REQUEST_DELAY)

        # De-duplicate by URL
        seen: set[str] = set()
        unique: list[dict] = []
        for r in all_rows:
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)

        logger.success(f"Catalogue listing complete: {len(unique)} unique assessments")
        return unique

    # ------------------------------------------------------------------ #

    def enrich_with_detail_pages(
        self, listings: list[dict], max_workers: int = 1
    ) -> list[dict]:
        """Visit each assessment's detail page for richer metadata."""
        enriched: list[dict] = []
        total = len(listings)
        for i, assessment in enumerate(listings, 1):
            logger.info(f"[{i}/{total}] Enriching: {assessment['name']}")
            enriched.append(_scrape_detail_page(self.session, dict(assessment)))
        return enriched

    # ------------------------------------------------------------------ #

    def scrape(self, enrich_details: bool = True) -> list[dict]:
        """
        Full pipeline:
        1. Scrape catalogue listings
        2. (Optional) enrich with detail pages
        3. Save to JSON
        """
        logger.info("Starting SHL catalogue scrape …")
        listings = self.scrape_catalogue_listings()

        if enrich_details:
            logger.info("Enriching assessments with detail-page data …")
            data = self.enrich_with_detail_pages(listings)
        else:
            data = listings

        self._save(data)
        return data

    # ------------------------------------------------------------------ #

    def _save(self, data: list[dict]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.success(
            f"Saved {len(data)} assessments → {self.output_path}"
        )

    # ------------------------------------------------------------------ #

    def load_cached(self) -> list[dict]:
        """Load previously scraped data from disk."""
        if not self.output_path.exists():
            raise FileNotFoundError(
                f"No cached data at {self.output_path}. Run scrape() first."
            )
        with open(self.output_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
