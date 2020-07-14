from typing import List, Tuple, Iterable
from pathlib import Path
import json
import xml.etree.ElementTree as ET

VALIDATION_FREQUENCY = 25


def prepare(clues: Iterable[Tuple[str, str]]):
    clues_dir = Path("tmp/clues")
    clues_dir.mkdir(parents=True, exist_ok=True)

    with open(clues_dir / "examples_train.txt", "a") as fet, open(clues_dir / "labels_train.txt", "a") as flt, open(
        clues_dir / "examples_validation.txt", "a"
    ) as fev, open(clues_dir / "labels_validation.txt", "a") as flv:
        for i, (example, label) in enumerate(clues):
            if i % VALIDATION_FREQUENCY == 0:
                fev.write(f"{example}\n")
                flv.write(f"{label}\n")
            else:
                fet.write(f"{example}\n")
                flt.write(f"{label}\n")


def prepare_all():
    prepare_guardian()
    prepare_independent()
    prepare_mirror()


def prepare_guardian():
    crosswords_dir = Path("crosswords")

    with open(crosswords_dir / "guardian.ldjson", "r") as f:
        prepare(parse_guardian(f))


def parse_guardian(crosswords: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for line in crosswords:
        for entry in json.loads(line)["entries"]:
            clue = entry["clue"].strip().replace("\r", "").replace("\n", "")
            solution = entry["solution"]
            group = entry["group"]
            if clue != "" and solution != "" and len(group) == 1:
                yield clue, solution


def prepare_independent():
    prepare_arkadium(Path("crosswords/independent"))


def prepare_mirror():
    prepare_arkadium(Path("crosswords/mirror"))


def prepare_arkadium(crosswords_dir: Path):
    crosswords: List[Tuple[str, str]] = []

    for crossword_file in crosswords_dir.iterdir():
        try:
            crosswords.extend(parse_arkadium(crossword_file.read_text()))
        except:
            print(f"Failed reading {crossword_file}")
            raise

    prepare(crosswords)


def parse_range(range_str: str) -> List[int]:
    return ([int(i) for i in range_str.split("-")] * 2)[:2]


ns = {"cc": "http://crossword.info/xml/crossword-compiler", "rp": "http://crossword.info/xml/rectangular-puzzle"}


def lookup_range(root: ET.Element, x_range: str, y_range: str) -> str:
    parsed_x_range = parse_range(x_range)
    parsed_y_range = parse_range(y_range)

    assert parsed_x_range[0] == parsed_x_range[1] or parsed_y_range[0] == parsed_y_range[1]

    answer = ""

    for x in range(parsed_x_range[0], parsed_x_range[1] + 1):
        for y in range(parsed_y_range[0], parsed_y_range[1] + 1):
            cell = root.find(f".//rp:cell[@x='{x}'][@y='{y}']", ns)
            assert cell is not None
            solution = cell.get("solution")
            assert solution is not None
            answer += solution

    return answer


def parse_arkadium(crossword: str) -> Iterable[Tuple[str, str]]:
    root = ET.fromstring(crossword)
    for clue in root.findall(".//rp:clue", ns):
        if clue.get("is-link") is not None:
            continue
        word = root.find(f".//rp:word[@id='{clue.get('word')}']", ns)
        assert word is not None
        x = word.get("x")
        assert x is not None
        y = word.get("y")
        assert y is not None
        answer = lookup_range(root, x, y)
        for cells in word.findall("rp:cells", ns):
            x = cells.get("x")
            assert x is not None
            y = cells.get("y")
            assert y is not None
            answer += lookup_range(root, x, y)
        if clue.text is None:
            continue
        normalised_clue = clue.text.replace("\r", "").replace("\n", "")
        yield f"{normalised_clue} ({clue.get('format')})", answer
