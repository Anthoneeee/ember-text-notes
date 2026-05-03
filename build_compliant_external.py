from __future__ import annotations

import csv
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


RAW_DIR = Path("deliverables/external_headlines/raw")
OUT_DIR = Path("deliverables/external_headlines/processed")
BASE_CSV = Path("deliverables/dataset/scraped_headlines_clean_headline_only.csv")
OUT_CSV = OUT_DIR / "external_headlines_compliant.csv"
TRAIN_CSV = OUT_DIR / "augmented_headlines_compliant.csv"


def normalize(text: object) -> str:
    s = html.unescape(str(text or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s, flags=re.I)
    s = re.sub(r"\b[\w.-]+\.(?:com|org|net|gov|edu|co|io)\b", " ", s, flags=re.I)
    s = re.sub(r"\b(?:rcna|ncna|nca|fnc|rcrd)\d+\b", " ", s, flags=re.I)
    s = re.sub(r"\s+[\-|]\s+(?:NBC News|Fox News|TODAY|MSNBC).*$", "", s, flags=re.I)
    s = re.sub(r"^(?:NBC News|Fox News|TODAY|MSNBC)\s*[\-|:]\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s.replace("\n", " ").replace("\t", " "))
    return s.strip()


def key(text: object) -> str:
    s = normalize(text).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def keep(text: str) -> bool:
    s = normalize(text)
    low = s.lower()
    if len(s) < 15 or len(s) > 220:
        return False
    if len(re.findall(r"[A-Za-z][A-Za-z']+", s)) < 3:
        return False
    if re.search(r"https?://|www\.|\b(?:urlpath|domain|host|section_|subsection_)", low):
        return False
    if re.search(r"\b(?:rcna|ncna|nca|fnc|rcrd)\d+\b", low):
        return False
    blocked = {
        "newsletter",
        "fox news",
        "nbc news",
        "breaking news",
        "latest news",
        "movies",
        "celebrity news",
        "entertainment media",
        "healthy foods",
        "nutrition",
        "white house",
    }
    if low in blocked:
        return False
    if any(phrase in low for phrase in ("access denied", "just a moment", "enable javascript")):
        return False
    return True


def walk_json(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_json(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_json(value)


def json_payloads(raw: str) -> list[object]:
    payloads: list[object] = []
    for match in re.finditer(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', raw, re.I | re.S):
        text = html.unescape(match.group(1)).strip()
        try:
            payloads.append(json.loads(text))
        except Exception:
            continue
    next_match = re.search(r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', raw, re.I | re.S)
    if next_match:
        try:
            payloads.append(json.loads(html.unescape(next_match.group(1)).strip()))
        except Exception:
            pass
    return payloads


def add(rows: list[dict[str, object]], source: str, label: int, headline: object, origin: str, url: str = "") -> None:
    h = normalize(headline)
    if keep(h):
        rows.append({"url": url, "source": source, "label": label, "headline": h, "origin": origin})


def parse_html(path: Path, source: str, label: int, rows: list[dict[str, object]]) -> None:
    raw = path.read_text(errors="ignore")
    origin = path.name
    for payload in json_payloads(raw):
        for node in walk_json(payload):
            if not isinstance(node, dict):
                continue
            for field in ("headline", "alternativeHeadline"):
                value = node.get(field)
                if isinstance(value, str):
                    add(rows, source, label, value, origin)
            alts = node.get("headlineAlternatives")
            if isinstance(alts, list):
                for alt in alts:
                    if isinstance(alt, dict) and isinstance(alt.get("text"), str):
                        add(rows, source, label, alt["text"], origin)

    for match in re.finditer(r'"headline"\s*:\s*"((?:\\.|[^"\\])*)"', raw):
        try:
            add(rows, source, label, json.loads('"' + match.group(1) + '"'), origin)
        except Exception:
            continue

    for match in re.finditer(r'title\s*:\s*"((?:\\.|[^"\\])*)"', raw):
        try:
            add(rows, source, label, json.loads('"' + match.group(1) + '"'), origin)
        except Exception:
            continue


def parse_rss(path: Path, source: str, label: int, rows: list[dict[str, object]]) -> None:
    try:
        root = ET.fromstring(path.read_text(errors="ignore"))
    except ET.ParseError:
        return
    for item in root.findall(".//item"):
        title_el = item.find("title")
        link_el = item.find("link")
        title = title_el.text if title_el is not None else ""
        url = link_el.text if link_el is not None else ""
        add(rows, source, label, title, path.name, url or "")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for path in sorted(RAW_DIR.glob("nbc_*.html")):
        parse_html(path, "NBC", 1, rows)
    for path in sorted(RAW_DIR.glob("nbc_feed_*.xml")):
        parse_rss(path, "NBC", 1, rows)
    for path in sorted(RAW_DIR.glob("fox_*.html")):
        parse_html(path, "FoxNews", 0, rows)
    for path in sorted(RAW_DIR.glob("fox_feed_*.xml")):
        parse_rss(path, "FoxNews", 0, rows)

    base = pd.read_csv(BASE_CSV)
    seen = {key(x) for x in base["headline"].astype(str)}
    unique = []
    for row in rows:
        k = key(row["headline"])
        if not k or k in seen:
            continue
        seen.add(k)
        unique.append(row)

    external = pd.DataFrame(unique, columns=["url", "source", "label", "headline", "origin"])
    external.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    base_out = base[["url", "source", "label", "headline"]].copy()
    train = pd.concat([base_out, external[["url", "source", "label", "headline"]]], ignore_index=True)
    train.to_csv(TRAIN_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    print("base_rows:", len(base_out))
    print("external_rows:", len(external))
    print("combined_rows:", len(train))
    if not external.empty:
        print("external_by_source:")
        print(external.groupby(["source", "label"]).size().to_string())
        print("top_origins:")
        print(external["origin"].value_counts().head(30).to_string())


if __name__ == "__main__":
    main()
