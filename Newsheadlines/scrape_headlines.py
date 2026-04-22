from __future__ import annotations

import argparse
import concurrent.futures as futures
import html
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Unified headers to reduce immediate anti-bot rejections.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


# Metadata priority: try generic meta tags first, then fall back to h1.
META_TITLE_SELECTORS: Tuple[Tuple[str, str], ...] = (
    ("property", "og:title"),
    ("name", "twitter:title"),
    ("name", "headline"),
    ("itemprop", "headline"),
)


@dataclass
class FetchResult:
    """Holds fetch result for a single URL."""

    url: str
    source: str
    label: Optional[int]
    success: bool
    headline: str
    status_code: Optional[int]
    error: str
    final_url: str
    elapsed_ms: int
    fetched_at_utc: str
    headline_method: str


def should_retry_http(status_code: int) -> bool:
    """Return whether HTTP status code is retryable."""
    return status_code in {403, 406, 408, 409, 425, 429, 500, 502, 503, 504}


def infer_source_and_label(url: str) -> Tuple[str, Optional[int]]:
    """Infer source and label from host name. Fox=0, NBC=1."""
    host = urlparse(url).netloc.lower()
    if "foxnews.com" in host:
        return "FoxNews", 0
    if "nbcnews.com" in host:
        return "NBC", 1
    return "Unknown", None


def clean_headline(text: str) -> str:
    """Clean headline text by trimming site suffixes and whitespace noise."""
    text = html.unescape(str(text))
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove common site suffixes to avoid polluting headline content.
    suffix_patterns = [
        r"\s*\|\s*Fox News\s*$",
        r"\s*-\s*Fox News\s*$",
        r"\s*\|\s*NBC News\s*$",
        r"\s*-\s*NBC News\s*$",
        r"\s*\|\s*NBC News Select\s*$",
    ]
    for pat in suffix_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    # Strip wrapping quote characters.
    text = text.strip("\"'“”‘’ ")
    return text


def extract_headline(soup: BeautifulSoup, source: str) -> str:
    """
    Extract headline:
    1) Try meta title fields first
    2) Apply site-specific h1 rules
    3) Fall back to the first h1
    """
    for attr, name in META_TITLE_SELECTORS:
        tag = soup.find("meta", attrs={attr: name})
        if tag and tag.get("content"):
            title = clean_headline(tag["content"])
            if title:
                return title

    # Per project example: Fox often uses h1 class similar to "headline speakable".
    if source == "FoxNews":
        fox_h1 = soup.find("h1", class_=re.compile(r"headline", flags=re.IGNORECASE))
        if fox_h1 and fox_h1.get_text(strip=True):
            return clean_headline(fox_h1.get_text(" ", strip=True))

    # Common NBC headline nodes.
    if source == "NBC":
        nbc_candidates = [
            "h1.article-hero-headline__htag",
            "h1[data-testid='headline']",
            "h1",
        ]
        for selector in nbc_candidates:
            node = soup.select_one(selector)
            if node and node.get_text(strip=True):
                return clean_headline(node.get_text(" ", strip=True))

    generic_h1 = soup.find("h1")
    if generic_h1 and generic_h1.get_text(strip=True):
        return clean_headline(generic_h1.get_text(" ", strip=True))

    return ""


def headline_from_url_slug(url: str) -> str:
    """
    Build fallback headline from URL slug when pages are not fetchable (e.g., 403/406).
    Note: this is not a true page-content extraction and is marked via headline_method.
    """
    parsed = urlparse(url)
    slug = parsed.path.strip("/")
    if not slug:
        return ""

    # Keep only the last slug segment to remove category-like prefixes.
    last = slug.split("/")[-1]
    last = re.sub(r"\.print$", "", last, flags=re.IGNORECASE)
    last = re.sub(r"\.html$", "", last, flags=re.IGNORECASE)
    last = last.replace("-", " ").replace("_", " ")
    last = clean_headline(last)
    return last


def generate_url_variants(url: str) -> List[str]:
    """
    Generate URL variants to recover from historical link-format issues
    (http/www/.print/trailing slash).
    """
    raw = str(url).strip()
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    path = parsed.path or "/"
    query = parsed.query or ""

    # Canonicalize host names.
    if host == "foxnews.com":
        host = "www.foxnews.com"
    if host == "nbcnews.com":
        host = "www.nbcnews.com"

    # Canonicalize scheme to https.
    canonical = urlunparse(("https", host, path, "", query, ""))

    variants: List[str] = [raw, canonical]

    # Variant without query string.
    no_query = urlunparse(("https", host, path, "", "", ""))
    variants.append(no_query)

    # Convert .print variants to canonical article URLs.
    if path.endswith(".print"):
        path_no_print = re.sub(r"\.print$", "", path, flags=re.IGNORECASE)
        variants.append(urlunparse(("https", host, path_no_print, "", query, "")))
        variants.append(urlunparse(("https", host, path_no_print, "", "", "")))

    # Try trailing-slash variants.
    if path != "/" and path.endswith("/"):
        path_no_slash = path.rstrip("/")
        variants.append(urlunparse(("https", host, path_no_slash, "", query, "")))
        variants.append(urlunparse(("https", host, path_no_slash, "", "", "")))
    elif path != "/" and not path.endswith("/"):
        path_with_slash = path + "/"
        variants.append(urlunparse(("https", host, path_with_slash, "", query, "")))

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for v in variants:
        vv = v.strip()
        if vv and vv not in seen:
            deduped.append(vv)
            seen.add(vv)
    return deduped


def fetch_html_headline(
    url: str,
    source: str,
    timeout_s: int,
    retries: int,
    min_delay_s: float,
    max_delay_s: float,
) -> Tuple[Optional[str], Optional[int], str, str, str]:
    """
    Stage 1: direct site fetch with URL variants.
    Returns: headline, status_code, error, final_url, method
    """
    variants = generate_url_variants(url)
    last_err = ""
    last_status: Optional[int] = None
    final_url = url

    for idx, variant in enumerate(variants):
        method = "direct_html" if idx == 0 else "direct_html_variant"

        for attempt in range(retries + 1):
            # Rate limiting to reduce anti-bot triggers.
            if max_delay_s > 0:
                delay = random.uniform(min_delay_s, max_delay_s)
                if delay > 0:
                    time.sleep(delay)

            try:
                resp = requests.get(
                    variant,
                    headers=HEADERS,
                    timeout=timeout_s,
                    allow_redirects=True,
                )
                last_status = resp.status_code
                final_url = str(resp.url)

                if resp.status_code != 200:
                    last_err = f"HTTP {resp.status_code}"
                    if should_retry_http(resp.status_code) and attempt < retries:
                        continue
                    break

                if "text/html" not in resp.headers.get("Content-Type", ""):
                    last_err = f"Unexpected content type: {resp.headers.get('Content-Type', '')}"
                    break

                soup = BeautifulSoup(resp.text, "html.parser")
                headline = extract_headline(soup, source)
                if headline:
                    return headline, last_status, "", final_url, method

                last_err = "Empty headline after parsing"
                break
            except Exception as ex:  # noqa: BLE001
                last_err = f"{type(ex).__name__}: {ex}"
                if attempt < retries:
                    continue
                break

    return None, last_status, last_err, final_url, "none"


def fetch_wayback_headline(
    url: str,
    source: str,
    timeout_s: int,
    min_delay_s: float,
    max_delay_s: float,
) -> Tuple[Optional[str], Optional[int], str, str]:
    """
    Stage 2: fetch headline from nearest Wayback snapshot.
    """
    archive_url = "https://web.archive.org/web/0/" + str(url).strip()
    if max_delay_s > 0:
        delay = random.uniform(min_delay_s, max_delay_s)
        if delay > 0:
            time.sleep(delay)
    try:
        resp = requests.get(
            archive_url,
            headers=HEADERS,
            timeout=max(timeout_s, 15),
            allow_redirects=True,
        )
        status = resp.status_code
        if status != 200:
            return None, status, f"Wayback HTTP {status}", archive_url
        soup = BeautifulSoup(resp.text, "html.parser")
        headline = extract_headline(soup, source)
        if headline:
            return headline, status, "", str(resp.url)
        return None, status, "Wayback empty headline", str(resp.url)
    except Exception as ex:  # noqa: BLE001
        return None, None, f"Wayback {type(ex).__name__}: {ex}", archive_url


def fetch_jina_headline(
    url: str,
    timeout_s: int,
    min_delay_s: float,
    max_delay_s: float,
) -> Tuple[Optional[str], Optional[int], str, str]:
    """
    Stage 3: use r.jina.ai mirror to extract title text.
    """
    mirror_url = "https://r.jina.ai/http://" + str(url).replace("https://", "").replace("http://", "")
    if max_delay_s > 0:
        delay = random.uniform(min_delay_s, max_delay_s)
        if delay > 0:
            time.sleep(delay)
    try:
        resp = requests.get(mirror_url, timeout=max(timeout_s, 15), allow_redirects=True)
        status = resp.status_code
        if status != 200:
            return None, status, f"Jina HTTP {status}", mirror_url
        m = re.search(r"^Title:\s*(.+?)\n", resp.text, flags=re.M)
        if m:
            title = clean_headline(m.group(1))
            if title:
                return title, status, "", str(resp.url)
        return None, status, "Jina empty title", str(resp.url)
    except Exception as ex:  # noqa: BLE001
        return None, None, f"Jina {type(ex).__name__}: {ex}", mirror_url


def fetch_one(
    url: str,
    timeout_s: int,
    retries: int,
    allow_url_fallback: bool,
    use_wayback: bool,
    use_jina: bool,
    min_delay_s: float,
    max_delay_s: float,
) -> FetchResult:
    """Fetch one URL with retry and fallback stages."""
    source, label = infer_source_and_label(url)
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    last_err = ""
    last_status: Optional[int] = None
    final_url = url

    # Stage 1: direct fetch (with URL variants)
    headline, status, err, final, method = fetch_html_headline(
        url=url,
        source=source,
        timeout_s=timeout_s,
        retries=retries,
        min_delay_s=min_delay_s,
        max_delay_s=max_delay_s,
    )
    if headline:
        elapsed = int((time.perf_counter() - start) * 1000)
        return FetchResult(
            url=url,
            source=source,
            label=label,
            success=True,
            headline=headline,
            status_code=status,
            error="",
            final_url=final,
            elapsed_ms=elapsed,
            fetched_at_utc=fetched_at_utc,
            headline_method=method,
        )
    last_status = status
    last_err = err
    final_url = final

    # Stage 2: Wayback
    if use_wayback:
        wb_headline, wb_status, wb_err, wb_final = fetch_wayback_headline(
            url=url,
            source=source,
            timeout_s=timeout_s,
            min_delay_s=min_delay_s,
            max_delay_s=max_delay_s,
        )
        if wb_headline:
            elapsed = int((time.perf_counter() - start) * 1000)
            return FetchResult(
                url=url,
                source=source,
                label=label,
                success=True,
                headline=wb_headline,
                status_code=wb_status,
                error="",
                final_url=wb_final,
                elapsed_ms=elapsed,
                fetched_at_utc=fetched_at_utc,
                headline_method="wayback_html",
            )
        last_status = wb_status if wb_status is not None else last_status
        last_err = wb_err or last_err
        final_url = wb_final or final_url

    # Stage 3: Jina mirror
    if use_jina:
        jina_headline, jina_status, jina_err, jina_final = fetch_jina_headline(
            url=url,
            timeout_s=timeout_s,
            min_delay_s=min_delay_s,
            max_delay_s=max_delay_s,
        )
        if jina_headline:
            elapsed = int((time.perf_counter() - start) * 1000)
            return FetchResult(
                url=url,
                source=source,
                label=label,
                success=True,
                headline=jina_headline,
                status_code=jina_status,
                error="",
                final_url=jina_final,
                elapsed_ms=elapsed,
                fetched_at_utc=fetched_at_utc,
                headline_method="jina_mirror",
            )
        last_status = jina_status if jina_status is not None else last_status
        last_err = jina_err or last_err
        final_url = jina_final or final_url

    elapsed = int((time.perf_counter() - start) * 1000)
    # Final fallback: generate headline from URL slug if page extraction fails.
    if allow_url_fallback:
        fallback = headline_from_url_slug(url)
        if fallback:
            return FetchResult(
                url=url,
                source=source,
                label=label,
                success=True,
                headline=fallback,
                status_code=last_status,
                error=last_err,
                final_url=final_url,
                elapsed_ms=elapsed,
                fetched_at_utc=fetched_at_utc,
                headline_method="url_slug_fallback",
            )

    return FetchResult(
        url=url,
        source=source,
        label=label,
        success=False,
        headline="",
        status_code=last_status,
        error=last_err,
        final_url=final_url,
        elapsed_ms=elapsed,
        fetched_at_utc=fetched_at_utc,
        headline_method="none",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch headlines for Project B URL list")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="Newsheadlines/url_only_data.csv",
        help="Input URL-list CSV (must contain a url column)",
    )
    parser.add_argument(
        "--output-raw",
        type=str,
        default="Newsheadlines/scraped_headlines_raw.csv",
        help="Output raw fetch-result CSV (includes failures)",
    )
    parser.add_argument(
        "--output-clean",
        type=str,
        default="Newsheadlines/scraped_headlines_clean.csv",
        help="Output cleaned CSV for training",
    )
    parser.add_argument("--timeout", type=int, default=15, help="Per-request timeout (seconds)")
    parser.add_argument("--retries", type=int, default=2, help="Retry count on failures")
    parser.add_argument("--max-workers", type=int, default=8, help="Thread pool worker count")
    parser.add_argument("--min-delay", type=float, default=0.0, help="Minimum pre-request delay (seconds)")
    parser.add_argument("--max-delay", type=float, default=0.0, help="Maximum pre-request delay (seconds)")
    parser.add_argument(
        "--disable-wayback",
        action="store_true",
        help="Disable Wayback fallback (enabled by default)",
    )
    parser.add_argument(
        "--disable-jina",
        action="store_true",
        help="Disable Jina mirror fallback (enabled by default)",
    )
    parser.add_argument(
        "--allow-url-fallback",
        action="store_true",
        help="Allow URL-slug fallback headline when page extraction fails",
    )
    parser.add_argument(
        "--remove-duplicate-urls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove duplicate URLs in cleaned output (default: enabled)",
    )
    parser.add_argument(
        "--remove-duplicate-headlines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove duplicate headlines in cleaned output (default: enabled)",
    )
    parser.add_argument(
        "--min-headline-chars",
        type=int,
        default=8,
        help="Drop headlines shorter than this length after cleanup (default: 8)",
    )
    parser.add_argument(
        "--drop-symbol-only-headlines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop headlines without any letters/digits (default: enabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "url" not in df.columns:
        raise ValueError(f"Input CSV must contain a 'url' column. Current columns: {df.columns.tolist()}")

    urls = [str(u).strip() for u in df["url"].tolist() if str(u).strip()]
    if not urls:
        raise ValueError("No usable URLs found in input CSV.")

    print(f"total_urls: {len(urls)}")
    print(f"max_workers: {args.max_workers}")
    print(f"timeout_s: {args.timeout}")
    print(f"retries: {args.retries}")
    print(f"min_delay_s: {args.min_delay}")
    print(f"max_delay_s: {args.max_delay}")
    print(f"use_wayback: {not args.disable_wayback}")
    print(f"use_jina: {not args.disable_jina}")
    print(f"allow_url_fallback: {args.allow_url_fallback}")
    print(f"remove_duplicate_urls: {args.remove_duplicate_urls}")
    print(f"remove_duplicate_headlines: {args.remove_duplicate_headlines}")
    print(f"min_headline_chars: {args.min_headline_chars}")
    print(f"drop_symbol_only_headlines: {args.drop_symbol_only_headlines}")

    results: List[FetchResult] = []
    completed = 0
    tic = time.perf_counter()

    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {
            pool.submit(
                fetch_one,
                u,
                args.timeout,
                args.retries,
                args.allow_url_fallback,
                not args.disable_wayback,
                not args.disable_jina,
                args.min_delay,
                args.max_delay,
            ): u
            for u in urls
        }
        for fut in futures.as_completed(future_map):
            res = fut.result()
            results.append(res)
            completed += 1

            # Print progress every 100 URLs to reduce noisy logs.
            if completed % 100 == 0 or completed == len(urls):
                ok = sum(1 for x in results if x.success)
                print(f"progress: {completed}/{len(urls)} | success: {ok}")

    elapsed_total = time.perf_counter() - tic
    raw_df = pd.DataFrame([r.__dict__ for r in results])
    raw_df = raw_df.sort_values("url").reset_index(drop=True)

    output_raw = Path(args.output_raw)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_raw, index=False)

    # Build cleaned dataset with explicit filtering + deduplication controls.
    clean_df = raw_df.copy()
    clean_input_rows = len(clean_df)

    clean_df = clean_df[clean_df["success"]].copy()
    removed_not_success = clean_input_rows - len(clean_df)

    clean_df = clean_df[clean_df["label"].notna()].copy()
    removed_no_label = clean_input_rows - removed_not_success - len(clean_df)
    clean_df["label"] = clean_df["label"].astype(int)

    clean_df["headline"] = clean_df["headline"].fillna("").astype(str).map(clean_headline)
    clean_df["url"] = clean_df["url"].fillna("").astype(str).str.strip()

    before_empty_filter = len(clean_df)
    clean_df = clean_df[clean_df["headline"].str.len() > 0].copy()
    removed_empty = before_empty_filter - len(clean_df)

    before_short_filter = len(clean_df)
    if args.min_headline_chars > 0:
        clean_df = clean_df[clean_df["headline"].str.len() >= args.min_headline_chars].copy()
    removed_short = before_short_filter - len(clean_df)

    before_symbol_filter = len(clean_df)
    if args.drop_symbol_only_headlines:
        clean_df = clean_df[
            clean_df["headline"].map(lambda x: any(ch.isalnum() for ch in str(x)))
        ].copy()
    removed_symbol_only = before_symbol_filter - len(clean_df)

    before_url_dedup = len(clean_df)
    if args.remove_duplicate_urls:
        clean_df["url_key"] = clean_df["url"].str.lower()
        clean_df = clean_df.drop_duplicates(subset=["url_key"], keep="first").copy()
    removed_dup_url = before_url_dedup - len(clean_df)

    before_headline_dedup = len(clean_df)
    if args.remove_duplicate_headlines:
        clean_df["headline_key"] = clean_df["headline"].str.lower()
        clean_df = clean_df.drop_duplicates(subset=["headline_key"], keep="first").copy()
    removed_dup_headline = before_headline_dedup - len(clean_df)

    # Keep only fields used by downstream training.
    clean_df = clean_df[["url", "source", "label", "headline"]].reset_index(drop=True)

    output_clean = Path(args.output_clean)
    output_clean.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_clean, index=False)

    total = len(raw_df)
    success_count = int(raw_df["success"].sum())
    fail_count = total - success_count
    method_counts = raw_df["headline_method"].value_counts(dropna=False).to_dict()

    print("----- summary -----")
    print(f"elapsed_total_s: {elapsed_total:.2f}")
    print(f"raw_total: {total}")
    print(f"raw_success: {success_count}")
    print(f"raw_fail: {fail_count}")
    print(f"headline_method_counts: {method_counts}")
    print(f"clean_rows: {len(clean_df)}")
    print(f"clean_removed_not_success: {removed_not_success}")
    print(f"clean_removed_no_label: {removed_no_label}")
    print(f"clean_removed_empty_headline: {removed_empty}")
    print(f"clean_removed_short_headline: {removed_short}")
    print(f"clean_removed_symbol_only_headline: {removed_symbol_only}")
    print(f"clean_removed_duplicate_urls: {removed_dup_url}")
    print(f"clean_removed_duplicate_headlines: {removed_dup_headline}")
    print(f"saved_raw: {output_raw.resolve()}")
    print(f"saved_clean: {output_clean.resolve()}")


if __name__ == "__main__":
    main()
