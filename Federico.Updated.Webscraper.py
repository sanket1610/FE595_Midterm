from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
from datetime import datetime, date, timedelta
from random import randint
import re


def extract_source(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
    }
    source = requests.get(url, headers=headers).text
    return source


def get_headlines(stock_name, num_pages, randvalue_begin, randvalue_end):
    df = pd.DataFrame()
    for page_index in range(1, num_pages):
        s = extract_source(
            "https://seekingalpha.com/symbol/"
            + stock_name
            + "/more_focus?page="
            + str(page_index)
            + "&new_layout=true"
        )

        # https://stackoverflow.com/questions/54496863/python-parse-html-with-escape-characters

        soup = BeautifulSoup(s.replace(r"\"", '"').replace(r"\/", "/"), "html.parser")
        df = df.append(bs_get(soup))
        time.sleep(randint(randvalue_begin, randvalue_end))
    return df


def format_date(x):
    if x.startswith("Today"):
        c = date.today()
    elif x.startswith("Yesterday"):
        c = date.today() - timedelta(days=1)
    else:
        match = re.search(r"(\D{3}\.\s*\d{1,2})$", x)
        if match is not None:
            x = match[0] + ", 2020"
        c = datetime.strptime(x, "%b. %d, %Y")
    c = c.strftime("%m/%d/%Y")
    return c


def bs_get(soup):
    container = soup.find_all("li", class_="symbol_item")
    headlines = []
    dates = []
    for con in container:
        headline = con.find("div", class_="symbol_article").a.get_text()
        headlines.append(headline)
        d = con.find("span", class_="date").get_text()
        d = format_date(d)
        dates.append(d)
    df = pd.DataFrame({"Headlines": headlines, "Dates": dates})
    return df


if __name__ == "__main__":

    stock_name = "TSLA"
    num_pages = 2
    randvalue_begin = 1
    randvalue_end = 15
    output_df = pd.DataFrame()
    output_filename = stock_name + datetime.now().strftime("_%Y%m%d_%H%M%S") + ".csv"
    output_df = get_headlines(stock_name, num_pages, randvalue_begin, randvalue_end)
    output_df.to_csv(output_filename)
