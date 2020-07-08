from typing import List, Tuple, Iterable
from pathlib import Path
import json

def read_guardian() -> List[Tuple[str, str]]:
    with open(Path("crosswords", "guardian.ldjson"), "r") as f:
        return parse_guardian(f)
    
def parse_guardian(crosswords: Iterable[str]) -> List[Tuple[str, str]]:
    return [
        (entry['clue'], entry['solution'])
        for line in crosswords
        for entry in json.loads(line)['entries']]