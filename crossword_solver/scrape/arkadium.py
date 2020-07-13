from datetime import timedelta, date
from typing import Iterator, Tuple, Callable
import requests
from pathlib import Path
import functools


def iterate_months(start_year: int, start_month: int, end_year: int, end_month: int) -> Iterator[Tuple[int, int]]:
    year = start_year
    month = start_month
    while year <= end_year:
        while month <= 12:
            yield year, month
            if month == end_month and year == end_year:
                return
            month += 1
        month = 1
        year += 1


def scrape_arkadium(
    name: str,
    create_url: Callable[[int, int, int], str],
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
):
    dir = Path("crosswords", name)
    dir.mkdir(parents=True, exist_ok=True)
    for year, month in iterate_months(start_year, start_month, end_year, end_month):
        for i in range(1, 100):
            url = create_url(year, month, i)
            print(f"Trying {url}")
            response = requests.get(url)
            if response.status_code != 200:
                break
            file_name = f"{year:04}_{month:02}_{i:02}.xml"
            (dir / file_name).write_text(response.text)
            print(f"Wrote crossword {file_name}")


def scrape_independent():
    scrape_arkadium(
        "independent",
        lambda year, month, i: f"https://ams.cdn.arkadiumhosted.com/assets/gamesfeed/independent/daily-crossword///c_{str(year)[-2:]}{month:02}{i:02}.xml",
        2015,
        6,
        2020,
        7,
    )


def scrape_mirror():
    scrape_arkadium(
        "mirror",
        lambda year, month, i: f"https://ams.cdn.arkadiumhosted.com/assets/gamesfeed/trinity-mirror-co-uk/mirror_crossword_feeds/cryptic_crossword//mir_2s_cryptic_{year}{month:02}{i:02}.xml",
        2017,
        6,
        2020,
        7,
    )
