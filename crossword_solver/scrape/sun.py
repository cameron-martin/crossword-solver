from datetime import timedelta, date
import itertools
from typing import Iterator
import requests
import json
from pathlib import Path

PUZZLE_ID_SEARCH_WINDOW = 100


def dates_backwards(from_date: date) -> Iterator[date]:
    for n in itertools.count(start=0):
        yield from_date - timedelta(days=n)


def scrape_sun():
    with open(Path("crosswords", "sun.ldjson"), "a") as f:
        current_puzzle_id = 46494
        for current_date in dates_backwards(date(year=2020, month=7, day=12)):
            found = False
            for candidate_puzzle_id in range(current_puzzle_id, current_puzzle_id - PUZZLE_ID_SEARCH_WINDOW, -1):
                url = f"https://feeds.thesun.co.uk/puzzles/crossword/{current_date.strftime('%Y%m%d')}/{candidate_puzzle_id}/data.json"
                print(f"Trying {url}")
                response = requests.get(url)
                if response.status_code == 200:
                    parsed_body = response.json()
                    if parsed_body["data"]["copy"]["crosswordtype"] == "Two Speed":
                        f.write(f"{json.dumps(parsed_body)}\n")
                        print(f"+ Puzzle found for day {current_date}")
                        found = True
                        current_puzzle_id = candidate_puzzle_id - 1
                        break
            if not found:
                print(f"- Puzzle not found for day {current_date}")
