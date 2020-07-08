import requests
import itertools
from bs4 import BeautifulSoup
from pathlib import Path

def scrape():
    with open(Path("crosswords", "guardian.ldjson"), "a") as f:
        for page_number in itertools.count(start=265):
            print(f"Scraping page number {page_number}")
            response = requests.get(f'https://www.theguardian.com/crosswords/series/cryptic?page={page_number}', allow_redirects=False)
            if response.status_code != 200:
                break
            soup = BeautifulSoup(response.text, "html.parser")
            crossword_links = set(x.get('href') for x in soup.find_all('a') if x.get('href') and x.get('href').startswith('https://www.theguardian.com/crosswords/cryptic/'))
            for crossword_link in crossword_links:
                print(f"  Scraping crossword {crossword_link}")
                response = requests.get(crossword_link)
                soup = BeautifulSoup(response.text, "html.parser")
                elem = soup.find(class_="js-crossword")
                if elem is None:
                    print(f"    Failed")
                    continue
                crossword = elem.get("data-crossword-data")
                f.write(f"{crossword}\n")
