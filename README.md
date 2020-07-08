# crossword-solver

Solves cryptic crosswords using ML.

## Usage

Crosswords can be scraped from the guardian website with the following command:

```sh
poetry run scrape
```

This creates a file `crosswords/guardian.ldjson`, which can then be consumed by other things (coming soon...). 