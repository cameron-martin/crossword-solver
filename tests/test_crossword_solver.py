from crossword_solver.read import parse_guardian


def test_parse_guardian():
    crosswords = [
        '''{"id":"crosswords/cryptic/28170","number":28170,"name":"Cryptic crossword No 28,170","creator":{"name":"Paul","webUrl":"https://www.theguardian.com/profile/paul"},"date":1593129600000,"entries":[{"id":"1-across","number":1,"humanNumber":"1","clue":"Pith extracted from some afore­mentioned tropical plant (6)","direction":"across","length":6,"group":["1-across"],"position":{"x":1,"y":0},"separatorLocations":{},"solution":"SESAME"},{"id":"10-across","number":10,"humanNumber":"10","clue":"Ultimately, Marilyn Monroe had to appear ditzy — never mind! (2,4,4)","direction":"across","length":10,"group":["10-across"],"position":{"x":5,"y":2},"separatorLocations":{",":[2,6]},"solution":"NOHARMDONE"}],"solutionAvailable":true,"dateSolutionAvailable":1593126000000,"dimensions":{"cols":15,"rows":15},"crosswordType":"cryptic","pdf":"https://crosswords-static.guim.co.uk/gdn.cryptic.20200626.pdf"}''',
        '''{"id":"crosswords/cryptic/28169","number":28169,"name":"Cryptic crossword No 28,169","creator":{"name":"Nutmeg","webUrl":"https://www.theguardian.com/profile/nutmeg"},"date":1593043200000,"entries":[{"id":"1-across","number":1,"humanNumber":"1","clue":"Denies one's corking drinks (7)","direction":"across","length":7,"group":["1-across"],"position":{"x":0,"y":0},"separatorLocations":{},"solution":"DISOWNS"},{"id":"5-across","number":5,"humanNumber":"5","clue":"Arrogantly ordered with knobs on? (6)","direction":"across","length":6,"group":["5-across"],"position":{"x":9,"y":0},"separatorLocations":{},"solution":"BOSSED"},{"id":"9-across","number":9,"humanNumber":"9","clue":"Capital fellow taking on jerk like Lothario? (8)","direction":"across","length":8,"group":["9-across"],"position":{"x":0,"y":2},"separatorLocations":{},"solution":"ROMANTIC"}],"solutionAvailable":true,"dateSolutionAvailable":1593039600000,"dimensions":{"cols":15,"rows":15},"crosswordType":"cryptic","pdf":"https://crosswords-static.guim.co.uk/gdn.cryptic.20200625.pdf"}'''
    ]
    clues = parse_guardian(crosswords)
    assert clues == [
        ('Pith extracted from some afore­mentioned tropical plant (6)', 'SESAME'),
        ('Ultimately, Marilyn Monroe had to appear ditzy — never mind! (2,4,4)', 'NOHARMDONE'),
        ('Denies one\'s corking drinks (7)', 'DISOWNS'),
        ('Arrogantly ordered with knobs on? (6)', 'BOSSED'),
        ('Capital fellow taking on jerk like Lothario? (8)', 'ROMANTIC')
    ]
