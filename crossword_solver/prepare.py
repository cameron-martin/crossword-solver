from typing import List, Tuple, Iterable
from pathlib import Path
import json

def prepare_guardian():
    crosswords_dir = Path("crosswords")
    crosswords_dir.mkdir(parents=True, exist_ok=True)

    with open(crosswords_dir / "guardian.ldjson", "r") as f, open(crosswords_dir / "examples.txt", "a") as fe, open(crosswords_dir / "labels.txt", "a") as fl:
        for example, label in parse_guardian(f):
            fe.write(f"{example}\n")
            fl.write(f"{label}\n")

    
def parse_guardian(crosswords: Iterable[str]) -> List[Tuple[str, str]]:
    examples = []
    for line in crosswords:
        for entry in json.loads(line)['entries']:
            clue = entry['clue'].strip().replace("\r", "").replace("\n", "")
            solution = entry['solution']
            if clue != "" and solution != "":
                examples.append((clue, solution))
    return examples