import re
import pandas as pd

from pathlib import Path
from random import random
from time import sleep
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from furl import furl
from selenium import webdriver


def store_result(meta, participants, content):
    transcript_path = Path("transcripts")
    path = transcript_path / "parsed" / meta["symbol"]
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(content, columns=["speaker", "q&a", "content"]).to_csv(
        path / "content.csv", index=False
    )
    pd.DataFrame(participants, columns=["type", "name"]).to_csv(
        path / "participants.csv", index=False
    )
    pd.Series(meta).to_csv(path / "earnings.csv")


def parse_html(html):
    soup = BeautifulSoup(html, "lxml")
    date_pattern = re.compile(r"(\d{2})-(\d{2})-(\d{2})")
    quarter_pattern = re.compile(r"(\bQ\d\b)")

    meta, participants, content = {}, [], []
    h1 = soup.find("h1", itemprop="headline")
    if h1 is None:
        return
    h1 = h1.text
    meta["company"] = h1[: h1.find("(")].strip()
    meta["symbol"] = h1[h1.find("(") + 1 : h1.find(")")]

    title = soup.find("div", class_="title")
    if title is None:
        return
    title = title.text
    print(title)
    match = date_pattern.search(title)
    if match:
        m, d, y = match.groups()
        meta["month"] = int(m)
        meta["day"] = int(d)
        meta["year"] = int(y)
    match = quarter_pattern.search(title)
    if match:
        meta["quarter"] = match.group(0)

    qa = 0
    speaker_types = ["Executives", "Analysts"]
    for header in [p.parent for p in soup.find_all("strong")]:
        text = header.text.strip()
        if text.lower().startswith("copyright"):
            continue
        elif text.lower().startswith("question-and"):
            qa = 1
            continue
        elif any([speaker_type in text for speaker_type in speaker_types]):
            for participant in header.find_next_siblings("p"):
                if participant.find("strong"):
                    break
                else:
                    participants.append([text, participant.text])
        else:
            p = []
            for participant in header.find_next_siblings("p"):
                if participant.find("strong"):
                    break
                else:
                    p.append(participant.text)
            content.append([header.text, qa, "\n".join(p)])
    return meta, participants, content


if __name__ == "__main__":
    sa_url = "https://seekingalpha.com/"
    driver = webdriver.Chrome(executable_path="02_earnings_calls/chromedriver.exe")

    page = 1
    while True:
        print(f"Page: {page}")
        url = f"{sa_url}/earnings/earnings-call-transcripts/{page}"
        driver.get(urljoin(base=sa_url, url=url))
        sleep(8 + (random() - 0.5) * 2)
        response = driver.page_source
        page += 1
        soup = BeautifulSoup(response, "lxml")
        links = soup.find_all(name="a", string=re.compile("Earnings Call Transcript"))
        links = links[:2]
        if len(links) == 0:
            break
        for link in links:
            transcript_url = link.attrs.get("href")
            article_url = furl(urljoin(sa_url, transcript_url)).add({"part": "single"})
            driver.get(article_url.url)
            sleep(5 + (random() - 0.5) * 2)
            html = driver.page_source
            result = parse_html(html)
            if result is not None:
                meta, participants, content = result
                meta["link"] = link
                store_result(meta, participants, content)
                sleep(8 + (random() - 0.5) * 2)
        if page > 1:
            break

    driver.close()
