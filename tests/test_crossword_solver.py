from crossword_solver.prepare import parse_guardian, parse_arkadium


def test_parse_guardian():
    crosswords = [
        """{"id":"crosswords/cryptic/28170","number":28170,"name":"Cryptic crossword No 28,170","creator":{"name":"Paul","webUrl":"https://www.theguardian.com/profile/paul"},"date":1593129600000,"entries":[{"id":"1-across","number":1,"humanNumber":"1","clue":"Pith extracted from some afore­mentioned tropical plant (6)","direction":"across","length":6,"group":["1-across"],"position":{"x":1,"y":0},"separatorLocations":{},"solution":"SESAME"},{"id":"10-across","number":10,"humanNumber":"10","clue":"Ultimately, Marilyn Monroe had to appear ditzy — never mind! (2,4,4)","direction":"across","length":10,"group":["10-across"],"position":{"x":5,"y":2},"separatorLocations":{",":[2,6]},"solution":"NOHARMDONE"}],"solutionAvailable":true,"dateSolutionAvailable":1593126000000,"dimensions":{"cols":15,"rows":15},"crosswordType":"cryptic","pdf":"https://crosswords-static.guim.co.uk/gdn.cryptic.20200626.pdf"}""",
        """{"id":"crosswords/cryptic/28169","number":28169,"name":"Cryptic crossword No 28,169","creator":{"name":"Nutmeg","webUrl":"https://www.theguardian.com/profile/nutmeg"},"date":1593043200000,"entries":[{"id":"1-across","number":1,"humanNumber":"1","clue":"Denies one's corking drinks (7)","direction":"across","length":7,"group":["1-across"],"position":{"x":0,"y":0},"separatorLocations":{},"solution":"DISOWNS"},{"id":"5-across","number":5,"humanNumber":"5","clue":"Arrogantly ordered with knobs on? (6)","direction":"across","length":6,"group":["5-across"],"position":{"x":9,"y":0},"separatorLocations":{},"solution":"BOSSED"},{"id":"9-across","number":9,"humanNumber":"9","clue":"Capital fellow taking on jerk like Lothario? (8)","direction":"across","length":8,"group":["9-across"],"position":{"x":0,"y":2},"separatorLocations":{},"solution":"ROMANTIC"}],"solutionAvailable":true,"dateSolutionAvailable":1593039600000,"dimensions":{"cols":15,"rows":15},"crosswordType":"cryptic","pdf":"https://crosswords-static.guim.co.uk/gdn.cryptic.20200625.pdf"}""",
    ]
    clues = parse_guardian(crosswords)
    assert list(clues) == [
        ("Pith extracted from some afore­mentioned tropical plant (6)", "SESAME"),
        ("Ultimately, Marilyn Monroe had to appear ditzy — never mind! (2,4,4)", "NOHARMDONE"),
        ("Denies one's corking drinks (7)", "DISOWNS"),
        ("Arrogantly ordered with knobs on? (6)", "BOSSED"),
        ("Capital fellow taking on jerk like Lothario? (8)", "ROMANTIC"),
    ]


def test_parse_arkadium():
    xml_str = """<?xml version="1.0" encoding="UTF-8"?>
    <crossword-compiler xmlns="http://crossword.info/xml/crossword-compiler"><rectangular-puzzle xmlns="http://crossword.info/xml/rectangular-puzzle" alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"><metadata><title></title><creator></creator><copyright></copyright><description></description></metadata><crossword><grid width="15" height="15"><grid-look numbering-scheme="normal" clue-square-divider-width="0.7"></grid-look><cell x="1" y="1" solution="C" number="1"></cell><cell x="1" y="2" solution="A" number="8"></cell><cell x="1" y="3" solution="T"></cell><cell x="1" y="4" solution="H" number="12"></cell><cell x="1" y="5" solution="O"></cell><cell x="1" y="6" solution="L" number="14"></cell><cell x="1" y="7" solution="I"></cell><cell x="1" y="8" solution="C" number="18"></cell><cell x="1" y="9" type="block"></cell><cell x="1" y="10" solution="R" number="21"></cell><cell x="1" y="11" solution="E"></cell><cell x="1" y="12" solution="D" number="25"></cell><cell x="1" y="13" solution="R"></cell><cell x="1" y="14" solution="U" number="28"></cell><cell x="1" y="15" solution="M"></cell><cell x="2" y="1" type="block"></cell><cell x="2" y="2" solution="D"></cell><cell x="2" y="3" type="block"></cell><cell x="2" y="4" solution="A"></cell><cell x="2" y="5" type="block"></cell><cell x="2" y="6" solution="A"></cell><cell x="2" y="7" type="block"></cell><cell x="2" y="8" solution="H"></cell><cell x="2" y="9" type="block"></cell><cell x="2" y="10" solution="U"></cell><cell x="2" y="11" type="block"></cell><cell x="2" y="12" solution="E"></cell><cell x="2" y="13" type="block"></cell><cell x="2" y="14" solution="T"></cell><cell x="2" y="15" type="block"></cell><cell x="3" y="1" solution="S" number="2"></cell><cell x="3" y="2" solution="H"></cell><cell x="3" y="3" solution="O"></cell><cell x="3" y="4" solution="R"></cell><cell x="3" y="5" solution="T"></cell><cell x="3" y="6" solution="S"></cell><cell x="3" y="7" type="block"></cell><cell x="3" y="8" solution="A" number="19"></cell><cell x="3" y="9" solution="B"></cell><cell x="3" y="10" solution="S"></cell><cell x="3" y="11" solution="E"></cell><cell x="3" y="12" solution="N"></cell><cell x="3" y="13" solution="T"></cell><cell x="3" y="14" solution="E"></cell><cell x="3" y="15" solution="E"></cell><cell x="4" y="1" type="block"></cell><cell x="4" y="2" solution="O"></cell><cell x="4" y="3" type="block"></cell><cell x="4" y="4" solution="P"></cell><cell x="4" y="5" type="block"></cell><cell x="4" y="6" solution="T"></cell><cell x="4" y="7" type="block"></cell><cell x="4" y="8" solution="R"></cell><cell x="4" y="9" type="block"></cell><cell x="4" y="10" solution="H"></cell><cell x="4" y="11" type="block"></cell><cell x="4" y="12" solution="A"></cell><cell x="4" y="13" type="block"></cell><cell x="4" y="14" type="block"></cell><cell x="4" y="15" type="block"></cell><cell x="5" y="1" solution="S" number="3"></cell><cell x="5" y="2" solution="C"></cell><cell x="5" y="3" solution="H"></cell><cell x="5" y="4" solution="O"></cell><cell x="5" y="5" solution="O"></cell><cell x="5" y="6" solution="L"></cell><cell x="5" y="7" solution="M"></cell><cell x="5" y="8" solution="A"></cell><cell x="5" y="9" solution="R"></cell><cell x="5" y="10" solution="M"></cell><cell x="5" y="11" type="block"></cell><cell x="5" y="12" solution="R" number="26"></cell><cell x="5" y="13" solution="U"></cell><cell x="5" y="14" solution="M" number="29"></cell><cell x="5" y="15" solution="P"></cell><cell x="6" y="1" type="block"></cell><cell x="6" y="2" type="block"></cell><cell x="6" y="3" type="block"></cell><cell x="6" y="4" solution="O"></cell><cell x="6" y="5" type="block"></cell><cell x="6" y="6" solution="Y"></cell><cell x="6" y="7" type="block"></cell><cell x="6" y="8" solution="C"></cell><cell x="6" y="9" type="block"></cell><cell x="6" y="10" solution="O"></cell><cell x="6" y="11" type="block"></cell><cell x="6" y="12" solution="I"></cell><cell x="6" y="13" type="block"></cell><cell x="6" y="14" solution="A"></cell><cell x="6" y="15" type="block"></cell><cell x="7" y="1" solution="M" number="4"></cell><cell x="7" y="2" solution="O" number="9"></cell><cell x="7" y="3" solution="U"></cell><cell x="7" y="4" solution="N"></cell><cell x="7" y="5" solution="T"></cell><cell x="7" y="6" type="block"></cell><cell x="7" y="7" solution="S" number="17"></cell><cell x="7" y="8" solution="T"></cell><cell x="7" y="9" solution="A"></cell><cell x="7" y="10" solution="R"></cell><cell x="7" y="11" solution="T"></cell><cell x="7" y="12" solution="I"></cell><cell x="7" y="13" solution="N"></cell><cell x="7" y="14" solution="G"></cell><cell x="7" y="15" type="block"></cell><cell x="8" y="1" type="block"></cell><cell x="8" y="2" solution="S"></cell><cell x="8" y="3" type="block"></cell><cell x="8" y="4" type="block"></cell><cell x="8" y="5" type="block"></cell><cell x="8" y="6" solution="G" number="15"></cell><cell x="8" y="7" type="block"></cell><cell x="8" y="8" solution="E"></cell><cell x="8" y="9" type="block"></cell><cell x="8" y="10" solution="E"></cell><cell x="8" y="11" type="block"></cell><cell x="8" y="12" type="block"></cell><cell x="8" y="13" type="block"></cell><cell x="8" y="14" solution="I"></cell><cell x="8" y="15" type="block"></cell><cell x="9" y="1" type="block"></cell><cell x="9" y="2" solution="C" number="10"></cell><cell x="9" y="3" solution="O"></cell><cell x="9" y="4" solution="B" number="13"></cell><cell x="9" y="5" solution="B"></cell><cell x="9" y="6" solution="L"></cell><cell x="9" y="7" solution="E"></cell><cell x="9" y="8" solution="R"></cell><cell x="9" y="9" solution="S"></cell><cell x="9" y="10" type="block"></cell><cell x="9" y="11" solution="P" number="24"></cell><cell x="9" y="12" solution="R" number="27"></cell><cell x="9" y="13" solution="I"></cell><cell x="9" y="14" solution="C"></cell><cell x="9" y="15" solution="E"></cell><cell x="10" y="1" type="block"></cell><cell x="10" y="2" solution="A"></cell><cell x="10" y="3" type="block"></cell><cell x="10" y="4" solution="U"></cell><cell x="10" y="5" type="block"></cell><cell x="10" y="6" solution="O"></cell><cell x="10" y="7" type="block"></cell><cell x="10" y="8" solution="A"></cell><cell x="10" y="9" type="block"></cell><cell x="10" y="10" solution="R" number="22"></cell><cell x="10" y="11" type="block"></cell><cell x="10" y="12" solution="O"></cell><cell x="10" y="13" type="block"></cell><cell x="10" y="14" type="block"></cell><cell x="10" y="15" type="block"></cell><cell x="11" y="1" solution="T" number="5"></cell><cell x="11" y="2" solution="R"></cell><cell x="11" y="3" solution="O"></cell><cell x="11" y="4" solution="T"></cell><cell x="11" y="5" type="block"></cell><cell x="11" y="6" solution="S" number="16"></cell><cell x="11" y="7" solution="A"></cell><cell x="11" y="8" solution="C"></cell><cell x="11" y="9" solution="R"></cell><cell x="11" y="10" solution="E"></cell><cell x="11" y="11" solution="C"></cell><cell x="11" y="12" solution="O"></cell><cell x="11" y="13" solution="E"></cell><cell x="11" y="14" solution="U" number="30"></cell><cell x="11" y="15" solution="R"></cell><cell x="12" y="1" type="block"></cell><cell x="12" y="2" type="block"></cell><cell x="12" y="3" type="block"></cell><cell x="12" y="4" solution="T"></cell><cell x="12" y="5" type="block"></cell><cell x="12" y="6" solution="S"></cell><cell x="12" y="7" type="block"></cell><cell x="12" y="8" solution="T"></cell><cell x="12" y="9" type="block"></cell><cell x="12" y="10" solution="G"></cell><cell x="12" y="11" type="block"></cell><cell x="12" y="12" solution="F"></cell><cell x="12" y="13" type="block"></cell><cell x="12" y="14" solution="S"></cell><cell x="12" y="15" type="block"></cell><cell x="13" y="1" solution="E" number="6"></cell><cell x="13" y="2" solution="L" number="11"></cell><cell x="13" y="3" solution="D"></cell><cell x="13" y="4" solution="O"></cell><cell x="13" y="5" solution="R"></cell><cell x="13" y="6" solution="A"></cell><cell x="13" y="7" solution="D"></cell><cell x="13" y="8" solution="O"></cell><cell x="13" y="9" type="block"></cell><cell x="13" y="10" solution="G" number="23"></cell><cell x="13" y="11" solution="A"></cell><cell x="13" y="12" solution="T"></cell><cell x="13" y="13" solution="E"></cell><cell x="13" y="14" solution="A"></cell><cell x="13" y="15" solution="U"></cell><cell x="14" y="1" type="block"></cell><cell x="14" y="2" solution="E"></cell><cell x="14" y="3" type="block"></cell><cell x="14" y="4" solution="C"></cell><cell x="14" y="5" type="block"></cell><cell x="14" y="6" solution="R"></cell><cell x="14" y="7" type="block"></cell><cell x="14" y="8" solution="R"></cell><cell x="14" y="9" type="block"></cell><cell x="14" y="10" solution="A"></cell><cell x="14" y="11" type="block"></cell><cell x="14" y="12" solution="O"></cell><cell x="14" y="13" type="block"></cell><cell x="14" y="14" solution="G"></cell><cell x="14" y="15" type="block"></cell><cell x="15" y="1" solution="J" number="7"></cell><cell x="15" y="2" solution="O"></cell><cell x="15" y="3" solution="C"></cell><cell x="15" y="4" solution="K"></cell><cell x="15" y="5" solution="E"></cell><cell x="15" y="6" solution="Y"></cell><cell x="15" y="7" type="block"></cell><cell x="15" y="8" solution="S" number="20"></cell><cell x="15" y="9" solution="T"></cell><cell x="15" y="10" solution="E"></cell><cell x="15" y="11" solution="E"></cell><cell x="15" y="12" solution="P"></cell><cell x="15" y="13" solution="L"></cell><cell x="15" y="14" solution="E"></cell><cell x="15" y="15" solution="D"></cell></grid><word id="1" x="1-5" y="2" solution="ad hoc"></word><word id="2" x="7-11" y="2"></word><word id="3" x="13-15" y="2"></word><word id="4" x="1-7" y="4"></word><word id="5" x="9-15" y="4"></word><word id="6" x="1-6" y="6"></word><word id="7" x="8-15" y="6"></word><word id="8" x="1-15" y="8" solution="character actors"></word><word id="9" x="10-15" y="10"></word><word id="10" x="1-7" y="12"></word><word id="11" x="9-15" y="12"></word><word id="12" x="1-3" y="14"></word><word id="13" x="5-9" y="14"></word><word id="14" x="11-15" y="14"></word><word id="15" x="1" y="1-8"></word><word id="16" x="5" y="1-10"></word><word id="17" x="7" y="1-5" solution="mount rushmore"><cells x="1-8" y="10"></cells></word><word id="18" x="11" y="1-4"></word><word id="19" x="13" y="1-8"></word><word id="20" x="15" y="1-6" solution="jockey shorts"><cells x="3" y="1-6"></cells></word><word id="21" x="9" y="2-9"></word><word id="22" x="11" y="6-15" solution="Sacre-Coeur"></word><word id="23" x="7" y="7-14" solution="starting price"><cells x="9" y="11-15"></cells></word><word id="24" x="3" y="8-15"></word><word id="25" x="15" y="8-15"></word><word id="26" x="1" y="10-15" solution="red rum"></word><word id="27" x="13" y="10-15"></word><word id="28" x="5" y="12-15"></word><clues ordering="normal"><title><b>Punk-8932...Across</b></title><clue word="1" number="8" format="2,3" citation="ADD HOCK HOM">For this is put on some horse, say?</clue><clue word="2" number="9" format="5" citation="O/SCAR">Love mark that&#39;s prized by luvvies</clue><clue word="3" number="11" format="3" citation="LE(g)O">Sign bricks no good</clue><clue word="4" number="12" format="7" citation="HARPO/ON">Sound marks on spear</clue><clue word="5" number="13" format="7" citation="BU(TTO&#60;)CK">Flip over the top during jump, one half on the saddle?</clue><clue word="6" number="14" format="6" citation="TA(S)LLY*">Second in tally misplaced, bringing up the rear</clue><clue word="7" number="15" format="8" citation="GLOSS(A/R)Y">A horse&#39;s third in photo finish - that explains everything!</clue><clue word="8" number="18" format="9,6" citation="CRY DEF">Those such as Dench playing M and Llewelyn playing Q?</clue><clue number="21" is-link="1" word="17">See 4 Down</clue><clue word="9" number="22" format="6" citation="RE(GG/A)E&#60;">Musical style always holding a horse back</clue><clue word="10" number="25" format="7" citation="DE/NARI/I&#60;">One country journalist sent over for money in the past</clue><clue word="11" number="27" format="7" citation="ROO/FT/OP">Native Australian newspaper getting work that may be slated?</clue><clue word="12" number="28" format="3" citation="HIDDEN">Transport in Oz commuter belts</clue><clue word="13" number="29" format="5" citation="DD">Very good spelling</clue><clue word="14" number="30" format="5" citation="U/S/AGE">Unemployment slowing initially, then time for employment</clue></clues><clues ordering="normal"><title><b>Down</b></title><clue word="15" number="1" format="8" citation="CAT-HOLIC">Liberal - one addicted to felines?</clue><clue number="2" is-link="1" word="20">See 7</clue><clue word="16" number="3" format="10" citation="S(C(H)OOL)MARM">Appear ingratiating welcoming Harrow&#39;s principal in aloof teacher</clue><clue word="17" number="4/21A" format="5,8" citation="CRY DEF">US memorial with words of encouragement from a 7 down?</clue><clue word="18" number="5" format="4" citation="DD">Pace of horse that&#39;s red</clue><clue word="19" number="6" format="8" citation="OLD DEAR*/O">Place of great riches where old dear unfortunately has nothing</clue><clue word="20" number="7/2" format="6,6" citation="JOCKEY (SHORT)S">Riders admitting their typical stature is pants</clue><clue word="21" number="10" format="8" citation="DD">Farriers for humans? Nonsense!</clue><clue word="22" number="16" format="5-5" citation="ANAG">Racecourse designed as a place of worship</clue><clue word="23" number="17/24" format="8,5" citation="ANAG ...and Arkle becomig SParkle with SP for starting price">Odds given by dodgy racing tipster - this making Arkle sparkle!</clue><clue word="24" number="19" format="8" citation="A/B(SENT)EE">Someone missing a worker, claiming dispatched</clue><clue word="25" number="20" format="8" citation="STEE(P(u)L(s)E)D">Pulse oddly found in horse built like a church?</clue><clue word="26" number="21" format="3,3" citation="REV">Horse butchery on the up</clue><clue word="27" number="23" format="6" citation="GA(TEA)U(l)">Frenchman endlessly quaffing drink that&#39;s sweet</clue><clue number="24" is-link="1" word="23">See 17</clue><clue word="28" number="26" format="4" citation="(t)RUMP">Bottom card, not top</clue></clues></crossword></rectangular-puzzle></crossword-compiler>
    """

    clues = list(parse_arkadium(xml_str))

    assert clues == [
        ("For this is put on some horse, say? (2,3)", "ADHOC"),
        ("Love mark that's prized by luvvies (5)", "OSCAR"),
        ("Sign bricks no good (3)", "LEO"),
        ("Sound marks on spear (7)", "HARPOON"),
        ("Flip over the top during jump, one half on the saddle? (7)", "BUTTOCK"),
        ("Second in tally misplaced, bringing up the rear (6)", "LASTLY"),
        ("A horse's third in photo finish - that explains everything! (8)", "GLOSSARY"),
        ("Those such as Dench playing M and Llewelyn playing Q? (9,6)", "CHARACTERACTORS"),
        ("Musical style always holding a horse back (6)", "REGGAE"),
        ("One country journalist sent over for money in the past (7)", "DENARII"),
        ("Native Australian newspaper getting work that may be slated? (7)", "ROOFTOP"),
        ("Transport in Oz commuter belts (3)", "UTE"),
        ("Very good spelling (5)", "MAGIC"),
        ("Unemployment slowing initially, then time for employment (5)", "USAGE"),
        ("Liberal - one addicted to felines? (8)", "CATHOLIC"),
        ("Appear ingratiating welcoming Harrow's principal in aloof teacher (10)", "SCHOOLMARM"),
        ("US memorial with words of encouragement from a 7 down? (5,8)", "MOUNTRUSHMORE"),
        ("Pace of horse that's red (4)", "TROT"),
        ("Place of great riches where old dear unfortunately has nothing (8)", "ELDORADO"),
        ("Riders admitting their typical stature is pants (6,6)", "JOCKEYSHORTS"),
        ("Farriers for humans? Nonsense! (8)", "COBBLERS"),
        ("Racecourse designed as a place of worship (5-5)", "SACRECOEUR"),
        ("Odds given by dodgy racing tipster - this making Arkle sparkle! (8,5)", "STARTINGPRICE"),
        ("Someone missing a worker, claiming dispatched (8)", "ABSENTEE"),
        ("Pulse oddly found in horse built like a church? (8)", "STEEPLED"),
        ("Horse butchery on the up (3,3)", "REDRUM"),
        ("Frenchman endlessly quaffing drink that's sweet (6)", "GATEAU"),
        ("Bottom card, not top (4)", "RUMP"),
    ]
