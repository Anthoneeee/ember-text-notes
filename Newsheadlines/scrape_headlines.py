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


# 统一请求头，降低被反爬直接拒绝的概率
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


# 元数据优先级：先尝试通用 meta，再回退到 h1
META_TITLE_SELECTORS: Tuple[Tuple[str, str], ...] = (
    ("property", "og:title"),
    ("name", "twitter:title"),
    ("name", "headline"),
    ("itemprop", "headline"),
)


@dataclass
class FetchResult:
    """保存单个 URL 的抓取结果。"""

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
    """定义可重试状态码。"""
    return status_code in {403, 406, 408, 409, 425, 429, 500, 502, 503, 504}


def infer_source_and_label(url: str) -> Tuple[str, Optional[int]]:
    """从域名推断新闻源与标签。Fox=0, NBC=1。"""
    host = urlparse(url).netloc.lower()
    if "foxnews.com" in host:
        return "FoxNews", 0
    if "nbcnews.com" in host:
        return "NBC", 1
    return "Unknown", None


def clean_headline(text: str) -> str:
    """做标题清洗，去掉站点尾缀与多余空白。"""
    text = html.unescape(str(text))
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 常见尾缀清理，避免把站点名当作标题内容
    suffix_patterns = [
        r"\s*\|\s*Fox News\s*$",
        r"\s*-\s*Fox News\s*$",
        r"\s*\|\s*NBC News\s*$",
        r"\s*-\s*NBC News\s*$",
        r"\s*\|\s*NBC News Select\s*$",
    ]
    for pat in suffix_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    # 清理包裹引号
    text = text.strip("\"'“”‘’ ")
    return text


def extract_headline(soup: BeautifulSoup, source: str) -> str:
    """
    提取标题：
    1) 先查 meta title
    2) 再走站点定制 h1 规则
    3) 最后回退第一个 h1
    """
    for attr, name in META_TITLE_SELECTORS:
        tag = soup.find("meta", attrs={attr: name})
        if tag and tag.get("content"):
            title = clean_headline(tag["content"])
            if title:
                return title

    # 按文档示例：Fox 常见 h1 class=headline speakable
    if source == "FoxNews":
        fox_h1 = soup.find("h1", class_=re.compile(r"headline", flags=re.IGNORECASE))
        if fox_h1 and fox_h1.get_text(strip=True):
            return clean_headline(fox_h1.get_text(" ", strip=True))

    # NBC 常见标题节点
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
    当站点返回 403/406 等不可抓取状态时，使用 URL slug 作为兜底标题。
    说明：该方法不是“页面正文抓取”，会在输出中标注 method 便于区分。
    """
    parsed = urlparse(url)
    slug = parsed.path.strip("/")
    if not slug:
        return ""

    # 去掉常见路径前缀（如 politics/sports/...），仅保留最后一段 slug
    last = slug.split("/")[-1]
    last = re.sub(r"\.print$", "", last, flags=re.IGNORECASE)
    last = re.sub(r"\.html$", "", last, flags=re.IGNORECASE)
    last = last.replace("-", " ").replace("_", " ")
    last = clean_headline(last)
    return last


def generate_url_variants(url: str) -> List[str]:
    """
    生成 URL 变体，尽量修复历史链接格式问题（http/www/.print/尾斜杠）。
    """
    raw = str(url).strip()
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    path = parsed.path or "/"
    query = parsed.query or ""

    # 统一站点主机名
    if host == "foxnews.com":
        host = "www.foxnews.com"
    if host == "nbcnews.com":
        host = "www.nbcnews.com"

    # 统一 https
    canonical = urlunparse(("https", host, path, "", query, ""))

    variants: List[str] = [raw, canonical]

    # 去掉 query 的版本
    no_query = urlunparse(("https", host, path, "", "", ""))
    variants.append(no_query)

    # .print 版本转原文版本
    if path.endswith(".print"):
        path_no_print = re.sub(r"\.print$", "", path, flags=re.IGNORECASE)
        variants.append(urlunparse(("https", host, path_no_print, "", query, "")))
        variants.append(urlunparse(("https", host, path_no_print, "", "", "")))

    # 尾斜杠变化
    if path != "/" and path.endswith("/"):
        path_no_slash = path.rstrip("/")
        variants.append(urlunparse(("https", host, path_no_slash, "", query, "")))
        variants.append(urlunparse(("https", host, path_no_slash, "", "", "")))
    elif path != "/" and not path.endswith("/"):
        path_with_slash = path + "/"
        variants.append(urlunparse(("https", host, path_with_slash, "", query, "")))

    # 去重保持顺序
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
    第一阶段：直连站点抓标题（含 URL 变体）。
    返回: headline, status_code, error, final_url, method
    """
    variants = generate_url_variants(url)
    last_err = ""
    last_status: Optional[int] = None
    final_url = url

    for idx, variant in enumerate(variants):
        method = "direct_html" if idx == 0 else "direct_html_variant"

        for attempt in range(retries + 1):
            # 限速：降低触发风控概率
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
    第二阶段：用 Wayback 最近快照抓真实标题。
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
    第三阶段：使用 r.jina.ai 镜像提取标题文本。
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
    """抓取单个 URL，失败时按重试次数重试。"""
    source, label = infer_source_and_label(url)
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    last_err = ""
    last_status: Optional[int] = None
    final_url = url

    # 阶段1：直连抓取（含 URL 变体）
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

    # 阶段2：Wayback
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

    # 阶段3：Jina 镜像
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
    # 最终兜底：若页面无法抓取，则尝试从 URL slug 生成标题
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
    parser = argparse.ArgumentParser(description="抓取 B 问 URL 列表对应的新闻标题")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="Newsheadlines/url_only_data.csv",
        help="输入 URL 列表 CSV，至少包含 url 列",
    )
    parser.add_argument(
        "--output-raw",
        type=str,
        default="Newsheadlines/scraped_headlines_raw.csv",
        help="输出原始抓取结果 CSV（含失败记录）",
    )
    parser.add_argument(
        "--output-clean",
        type=str,
        default="Newsheadlines/scraped_headlines_clean.csv",
        help="输出清洗后可训练 CSV",
    )
    parser.add_argument("--timeout", type=int, default=15, help="单次请求超时时间（秒）")
    parser.add_argument("--retries", type=int, default=2, help="失败重试次数")
    parser.add_argument("--max-workers", type=int, default=8, help="并发线程数")
    parser.add_argument("--min-delay", type=float, default=0.0, help="每次请求前最小延时（秒）")
    parser.add_argument("--max-delay", type=float, default=0.0, help="每次请求前最大延时（秒）")
    parser.add_argument(
        "--disable-wayback",
        action="store_true",
        help="禁用 Wayback 回退（默认开启）",
    )
    parser.add_argument(
        "--disable-jina",
        action="store_true",
        help="禁用 Jina 镜像回退（默认开启）",
    )
    parser.add_argument(
        "--allow-url-fallback",
        action="store_true",
        help="网页抓取失败时，允许从 URL slug 生成兜底标题",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_csv}")

    df = pd.read_csv(input_csv)
    if "url" not in df.columns:
        raise ValueError(f"输入 CSV 必须包含 url 列，当前列为: {df.columns.tolist()}")

    urls = [str(u).strip() for u in df["url"].tolist() if str(u).strip()]
    if not urls:
        raise ValueError("输入 CSV 中没有可用 URL。")

    print(f"total_urls: {len(urls)}")
    print(f"max_workers: {args.max_workers}")
    print(f"timeout_s: {args.timeout}")
    print(f"retries: {args.retries}")
    print(f"min_delay_s: {args.min_delay}")
    print(f"max_delay_s: {args.max_delay}")
    print(f"use_wayback: {not args.disable_wayback}")
    print(f"use_jina: {not args.disable_jina}")
    print(f"allow_url_fallback: {args.allow_url_fallback}")

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

            # 每 100 条输出一次进度，避免刷屏
            if completed % 100 == 0 or completed == len(urls):
                ok = sum(1 for x in results if x.success)
                print(f"progress: {completed}/{len(urls)} | success: {ok}")

    elapsed_total = time.perf_counter() - tic
    raw_df = pd.DataFrame([r.__dict__ for r in results])
    raw_df = raw_df.sort_values("url").reset_index(drop=True)

    output_raw = Path(args.output_raw)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_raw, index=False)

    # 生成清洗后数据：仅保留成功记录 + 非空标题 + 有效标签，并去重
    clean_df = raw_df[(raw_df["success"]) & (raw_df["headline"].str.len() > 0)].copy()
    clean_df = clean_df[clean_df["label"].notna()].copy()
    clean_df["label"] = clean_df["label"].astype(int)

    before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["headline", "label"], keep="first").copy()
    removed_dup = before_dedup - len(clean_df)

    # 仅保留训练常用字段
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
    print(f"clean_removed_duplicates: {removed_dup}")
    print(f"saved_raw: {output_raw.resolve()}")
    print(f"saved_clean: {output_clean.resolve()}")


if __name__ == "__main__":
    main()
