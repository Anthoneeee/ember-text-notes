import re

import pandas as pd
import requests


def norm(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def jina_title(url: str):
    try:
        jr = requests.get(
            "https://r.jina.ai/http://" + str(url).replace("https://", "").replace("http://", ""),
            timeout=20,
        )
        if jr.status_code != 200:
            return None
        m = re.search(r"^Title:\s*(.+?)\n", jr.text, flags=re.M)
        if m:
            return m.group(1).strip()
        return None
    except Exception:
        return None


def main():
    path = "/Users/anthony/PycharmProjects/pythonProject1/CIS5190/Newsheadlines/scraped_headlines_raw.csv"
    df = pd.read_csv(path)
    sample = df[df["headline_method"] == "direct_html"].sample(30, random_state=7)

    have = 0
    exact_match = 0
    for _, row in sample.iterrows():
        jt = jina_title(row["url"])
        if not jt:
            continue
        have += 1
        if norm(jt) == norm(row["headline"]):
            exact_match += 1

    print("sample", len(sample))
    print("jina_returned", have)
    print("exact_norm_match", exact_match)


if __name__ == "__main__":
    main()
