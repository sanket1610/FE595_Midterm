from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
from datetime import datetime, date, timedelta
import re
from random import shuffle, randint


def extract_source(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36(KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36"
    }
    source = requests.get(url, headers=headers).text
    return source


def get_headlines(ticker):
    df = pd.DataFrame()
    index = range(1, 15)
    shuffle(list(index))
    for i in index:
        s = extract_source(
            f"https://seekingalpha.com/symbol/{ticker}/more_focus?page={i}&new_layout=true"
        )
        # https://stackoverflow.com/questions/54496863/python-parse-html-with-escape-characters
        soup = BeautifulSoup(s.replace(r"\"", '"').replace(r"\/", "/"), "html.parser")
        print(soup)
        df = df.append(bs_get(soup))
        time.sleep(randint(3, 10))
    return df


def format_date(d):
    if d.startswith("Today"):
        ret = date.today()
    elif d.startswith("Yesterday"):
        ret = date.today() - timedelta(days=1)
    else:
        match = re.search(r"(\D{3}\.\ \d{1,2})$", d)
        if match is not None:  # the article is from this year
            d = match[0] + ", 2020"
        if "May" in d:  # may is not abbreviated so date has no "." after month
            ret = datetime.strptime(d, "%b %d, %Y")
        else:
            ret = datetime.strptime(d, "%b. %d, %Y")
    ret = ret.strftime("%-m/%-d/%Y")
    return ret


def bs_get(soup):
    container = soup.find_all("li", class_="symbol_item")
    headlines = []
    dates = []
    for con in container:
        headline = con.find("div", class_="symbol_article").a.get_text()
        headlines.append(headline)
        d = con.find("span", class_="date").get_text()
        dates.append(format_date(d))
    df = pd.DataFrame({"Headlines": headlines, "Dates": dates})
    return df


if __name__ == "__main__":
    ticker = input("Enter ticker to generate headlines: ")
    df = get_headlines(ticker)
    df.to_csv(f"{ticker}.csv", index=False)
