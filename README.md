# crossword-solver

Solves cryptic crosswords using ML.

## Usage

Install [poetry](https://python-poetry.org/).

Install dependencies:

```sh
poetry install
```

Crosswords can be scraped from the guardian website with the following command:

```sh
poetry run scrape
```

This creates a file `crosswords/guardian.ldjson`. This can then be converted into examples, `crosswords/examples.txt` and `crosswords/labels.txt`:

```sh
poetry run prepare
```

Training can then be run as the following:

```sh
poetry run train
```

For training using a GPU in docker, the following script can be run:

```sh
./bin/train-in-docker.sh
```