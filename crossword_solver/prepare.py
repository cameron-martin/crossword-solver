from typing import List, Tuple, Iterable
from pathlib import Path
import json

VALIDATION_FREQUENCY = 25


def prepare_guardian():
    crosswords_dir = Path("crosswords")
    crosswords_dir.mkdir(parents=True, exist_ok=True)

    with open(crosswords_dir / "guardian.ldjson", "r") as f, open(
        crosswords_dir / "examples_train.txt", "a"
    ) as fet, open(crosswords_dir / "labels_train.txt", "a") as flt, open(
        crosswords_dir / "examples_validation.txt", "a"
    ) as fev, open(
        crosswords_dir / "labels_validation.txt", "a"
    ) as flv:
        for i, (example, label) in enumerate(parse_guardian(f)):
            if i % VALIDATION_FREQUENCY == 0:
                fev.write(f"{example}\n")
                flv.write(f"{label}\n")
            else:
                fet.write(f"{example}\n")
                flt.write(f"{label}\n")


def parse_guardian(crosswords: Iterable[str]) -> List[Tuple[str, str]]:
    examples = []
    for line in crosswords:
        for entry in json.loads(line)["entries"]:
            clue = entry["clue"].strip().replace("\r", "").replace("\n", "")
            solution = entry["solution"]
            group = entry["group"]
            if clue != "" and solution != "" and len(group) == 1:
                examples.append((clue, solution))
    return examples
