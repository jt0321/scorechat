"""
commentary/anchoring.py
Which sonata, and which movement, a passage of commentary is about.

A passage's sonata comes from two witnesses. The manifest *declares* that a
section of the book discusses a sonata; the passage itself may *name* one -- an
opus number, a nickname, "the F-minor Sonata". The two are compared rather than
either being trusted alone:

    confirmed    declared, and the passage names that sonata too
    declared     declared, and the passage names nothing to check it against
    conflict     declared, but the passage names only other sonatas -- often a
                 comparison ("unlike Op. 57 ..."), sometimes a misfiled page;
                 kept under its declared sonata and flagged for review
    named        not declared, and the passage names exactly one sonata
    unanchored   not declared, and it names none, or several

Bar numbers are not an anchor. Commentators cite bars from their own editions
if at all, and nothing in the viewer or the chat may depend on them agreeing
with ours; what matters is the right work, the right movement, and at best the
general place in it.

A movement is found from cues the commentators actually use: a tempo marking
that only one movement of the sonata has ("Presto agitato"), a movement type
("the Scherzo", "the Rondo"), or an ordinal ("the second movement", "the
finale"). These books walk through a sonata movement by movement, so a cue
carries forward to the passages after it until the next one. A passage that
spans a change of movement is recorded against both.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --- the catalogue ----------------------------------------------------------------


@dataclass(frozen=True)
class Movement:
    work_id: int
    number: int
    tempo: str


@dataclass(frozen=True)
class Sonata:
    number: int                      # catalogue number, 1-32
    opus: int
    opus_sub: int | None             # the "No." within an opus
    label: str                       # "Op. 27 No. 2", as stored
    key: str                         # "C-sharp minor"
    nickname: str | None
    movements: tuple[Movement, ...]


class Catalogue:
    def __init__(self, sonatas: list[Sonata]):
        self.sonatas = {s.number: s for s in sonatas}

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "Catalogue":
        """From works rows: work_number, opus, nickname, key_signature,
        movement_number, tempo_indication, id."""
        grouped: dict[int, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["work_number"], []).append(row)
        sonatas = []
        for number, movements in sorted(grouped.items()):
            movements.sort(key=lambda r: r["movement_number"] or 0)
            first = movements[0]
            opus, sub = _parse_opus_label(first["opus"])
            sonatas.append(Sonata(
                number=number, opus=opus, opus_sub=sub, label=first["opus"],
                key=first["key_signature"] or "", nickname=first["nickname"],
                movements=tuple(Movement(r["id"], r["movement_number"], r["tempo_indication"] or "")
                                for r in movements)))
        return cls(sonatas)

    def by_opus(self, opus: int, sub: int | None = None) -> frozenset[int]:
        within = {s.number for s in self.sonatas.values() if s.opus == opus}
        if sub is not None:
            exact = {n for n in within if self.sonatas[n].opus_sub == sub}
            if exact:
                return frozenset(exact)
            # "Op. 2, 7, 10" is a list of opus numbers, not Op. 2 No. 7: a
            # number that does not exist falls back to the opus as a whole.
        return frozenset(within)

    def by_key(self, key: str) -> frozenset[int]:
        return frozenset(n for n, s in self.sonatas.items() if s.key.lower() == key.lower())


def _parse_opus_label(label: str) -> tuple[int, int | None]:
    found = re.match(r"Op\.\s*(\d+)\w*(?:\s+No\.\s*(\d+))?", label)
    return int(found.group(1)), (int(found.group(2)) if found.group(2) else None)


# --- what a passage names -----------------------------------------------------------

@dataclass(frozen=True)
class Mention:
    text: str
    kind: str                        # "opus" | "nickname" | "key"
    candidates: frozenset[int]
    start: int

    @property
    def unambiguous(self) -> bool:
        return len(self.candidates) == 1


# Nicknames by which the commentators name a sonata. "Quasi una fantasia" is
# Beethoven's own title for both of Op. 27, so it narrows to two, not one.
NICKNAMES = {
    "pathetique": {8}, "moonlight": {14}, "appassionata": {23}, "waldstein": {21},
    "hammerklavier": {29}, "les adieux": {26}, "adieux": {26}, "pastoral": {15},
    "pastorale": {15}, "tempest": {17}, "therese": {24},
    "quasi una fantasia": {13, 14}, "fantasie sonata": {13, 14}, "fantasia sonata": {13, 14},
}

# OCR renders digits in opus numbers as letters: "OP. loi" is Op. 101,
# "OP. io6" is Op. 106, "OP. Ill" is Op. 111, "Op. i4" is Op. 14.
_OCR_DIGITS = str.maketrans({"l": "1", "I": "1", "i": "1", "o": "0", "O": "0"})
_OCR_WHOLE = {"no": 110}             # "OP. no, A FLAT" is Op. 110

_OPUS = re.compile(
    r"\b[Oo0][Pp][.,]?\s*(?P<num>[0-9lIioO]{1,3}|no)(?![\w])a?"
    r"(?:[.,]?\s*(?:Nos?[.,]?\s*)?(?P<sub>[0-9Ii])(?![\w/]))?")
_KEY = (r"(?P<tonic>(?-i:[A-G]))(?:\s*[-‐]?\s*(?P<acc>sharp|flat|♯|♭|#))?"
        r"\s*[-‐]?\s*(?P<mode>major|minor)")
_KEYED_SONATA = re.compile(rf"\b{_KEY}\s+Sonata\b|\bSonata\s+in\s+{_KEY.replace('?P<', '?P<k2')}",
                           re.IGNORECASE)


def fold(text: str) -> str:
    """Lower-case ASCII, one character for one: "Pathétique" -> "pathetique".

    Length-preserving on purpose -- matches found in the folded text are quoted
    from the original by offset, and dropping a character would shift every
    quote after it.
    """
    return "".join((unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode() or " ")[0]
                   for ch in text).lower()


def key_name(tonic: str, accidental: str | None, mode: str) -> str:
    """Spell a key as the database does: "C-sharp minor", "E-flat major"."""
    accidental = {"#": "sharp", "♯": "sharp", "♭": "flat"}.get(accidental or "", accidental or "")
    tonic = tonic.upper() + (f"-{accidental.lower()}" if accidental else "")
    return f"{tonic} {mode.lower()}"


def find_mentions(text: str, catalogue: Catalogue) -> list[Mention]:
    mentions: list[Mention] = []
    for found in _OPUS.finditer(text):
        raw = found.group("num")
        number = _OCR_WHOLE.get(raw) or int(raw.translate(_OCR_DIGITS))
        sub = found.group("sub")
        candidates = catalogue.by_opus(number, int(sub.translate(_OCR_DIGITS)) if sub else None)
        if candidates:
            mentions.append(Mention(found.group(0), "opus", candidates, found.start()))

    folded = fold(text)
    for name, sonatas in NICKNAMES.items():
        for found in re.finditer(rf"\b{re.escape(name)}\b", folded):
            original = text[found.start():found.end()]
            # Several nicknames are ordinary words too: Op. 2 No. 2's Largo is
            # marked "appassionato", storms are "tempests". A name is a name
            # only when it is written as one.
            if original[:1].isupper():
                mentions.append(Mention(original, "nickname", frozenset(sonatas), found.start()))

    for found in _KEYED_SONATA.finditer(text):
        groups = found.groupdict()
        tonic = groups["tonic"] or groups["k2tonic"]
        key = key_name(tonic, groups["acc"] or groups["k2acc"], groups["mode"] or groups["k2mode"])
        candidates = catalogue.by_key(key)
        if candidates:
            mentions.append(Mention(found.group(0), "key", candidates, found.start()))

    return _narrow(sorted(mentions, key=lambda m: m.start))


def _narrow(mentions: list[Mention]) -> list[Mention]:
    """A key and an opus written together identify one sonata where neither
    does alone: "F-Major Sonata, Op. 10" is Op. 10 No. 2."""
    narrowed = []
    for mention in mentions:
        if mention.kind == "opus" and not mention.unambiguous:
            beside = [m for m in mentions if m.kind == "key" and 0 <= mention.start - m.start < 40]
            for key in beside:
                both = mention.candidates & key.candidates
                if len(both) == 1:
                    mention = Mention(key.text + " … " + mention.text, "opus", both, key.start)
        narrowed.append(mention)
    return narrowed


# --- which movement ---------------------------------------------------------------

@dataclass(frozen=True)
class Cue:
    movement: Movement
    text: str
    start: int
    kind: str                         # "tempo" | "type" | "ordinal"


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_TYPES = ("scherzo", "minuetto", "menuetto", "minuet", "trio", "rondo", "arietta",
          "introduzione", "introduction", "fuga", "fugue", "marcia", "march")
_TYPE_IN_TEMPO = {"minuet": ("minuetto", "menuetto", "minuetto", "tempo di menuetto", "tempo di minuetto"),
                  "trio": ("minuetto", "menuetto", "scherzo"),
                  "fugue": ("fuga",), "introduction": ("introduzione",), "march": ("marcia",)}


def tempo_phrase(tempo: str) -> str:
    """The tempo marking proper, without the movement title or metronome mark:
    "SCHERZO Allegretto" -> "allegretto", "Rondo: Allegro" -> "allegro"."""
    words = re.sub(r"\[.*?\]\s*=?\s*\d*", " ", tempo)
    words = re.sub(r"^\s*(?:[A-Z]{3,}\b|[A-Za-z' ]+:)\s*", "", words)
    return re.sub(r"[^a-z ]", " ", fold(words)).split("  ")[0].strip()


def find_cues(text: str, sonata: Sonata) -> list[Cue]:
    if not sonata.movements:
        return []
    folded = fold(text)
    cues: list[Cue] = []

    phrases = {m.work_id: tempo_phrase(m.tempo) for m in sonata.movements}
    for movement in sonata.movements:
        phrase = phrases[movement.work_id]
        if not phrase:
            continue
        # The full marking, and its first word alone -- commentators write "the
        # adagio" -- each usable only if no other movement's marking opens the
        # same way: "Allegro" beside "Allegro vivace" picks out neither.
        labels = [phrase] + ([phrase.split()[0]] if " " in phrase else [])
        found_any = False
        for label in labels:
            if found_any or any(other != movement.work_id and re.match(rf"{re.escape(label)}\b", p)
                                for other, p in phrases.items()):
                continue
            for found in re.finditer(rf"\b{re.escape(label)}\b", folded):
                found_any = True
                cues.append(Cue(movement, text[found.start():found.end()], found.start(), "tempo"))

    for word in _TYPES:
        for found in re.finditer(rf"\b{word}\b", folded):
            wanted = _TYPE_IN_TEMPO.get(word, (word,))
            matching = [m for m in sonata.movements if any(w in fold(m.tempo) for w in wanted)]
            if len(matching) == 1:
                cues.append(Cue(matching[0], text[found.start():found.end()], found.start(), "type"))

    last = sonata.movements[-1]
    for found in re.finditer(r"\b(first|second|third|fourth|last|final)\s+movement\b|\bfinale\b", folded):
        word = found.group(1)
        number = last.number if word in (None, "last", "final") else _ORDINALS[word]
        movement = next((m for m in sonata.movements if m.number == number), None)
        if movement:
            cues.append(Cue(movement, text[found.start():found.end()], found.start(), "ordinal"))

    return sorted(cues, key=lambda c: c.start)


# --- the anchor ------------------------------------------------------------------

@dataclass
class Anchor:
    status: str
    sonata: int | None
    work_ids: tuple[int, ...]
    evidence: dict = field(default_factory=dict)


@dataclass
class AnchorState:
    """What carries from one passage to the next within a section."""
    sonata: int | None = None
    movement: Movement | None = None


def anchor(text: str, declared: tuple[int, ...], catalogue: Catalogue, state: AnchorState) -> Anchor:
    mentions = find_mentions(text, catalogue)
    named = {n for m in mentions if m.unambiguous for n in m.candidates}
    evidence: dict = {"mentions": [{"text": m.text, "kind": m.kind, "sonatas": sorted(m.candidates)}
                                   for m in mentions]}

    if declared:
        consistent = [m for m in mentions if m.candidates & set(declared)]
        if consistent:
            status = "confirmed"
        elif named:
            status = "conflict"
            evidence["names_instead"] = sorted(named)
        else:
            status = "declared"
        sonata = _declared_sonata(declared, consistent, state)
    else:
        status = "named" if len(named) == 1 else "unanchored"
        sonata = next(iter(named)) if len(named) == 1 else None

    if sonata != state.sonata:
        state.sonata, state.movement = sonata, None
    work_ids: tuple[int, ...] = ()
    if sonata is not None:
        work_ids = _movements(text, catalogue.sonatas[sonata], state, evidence)
    return Anchor(status, sonata, work_ids, evidence)


def _declared_sonata(declared, consistent, state) -> int | None:
    if len(declared) == 1:
        return declared[0]
    # A section covering two sonatas (Elterlein's "Op. 14, Nos. 1 and 2"): a
    # passage belongs to one only when it picks it out, and the choice carries.
    picked = {n for m in consistent for n in m.candidates if n in declared}
    if len(picked) == 1:
        return next(iter(picked))
    return state.sonata if state.sonata in declared else None


def _movements(text: str, sonata: Sonata, state: AnchorState, evidence: dict) -> tuple[int, ...]:
    cues = find_cues(text, sonata)
    first_sentence_end = re.search(r"[.!?](\s|$)", text)
    opening = first_sentence_end.end() if first_sentence_end else len(text)

    start = state.movement
    if cues and cues[0].start < opening:
        start = cues[0].movement          # the passage opens by introducing it
    discussed = ([start] if start else []) + [c.movement for c in cues]
    if cues:
        state.movement = cues[-1].movement
    evidence["movement_cues"] = [{"text": c.text, "kind": c.kind, "movement": c.movement.number}
                                 for c in cues]
    if start and (not cues or cues[0].movement != start or cues[0].start >= opening):
        evidence["carried_from_previous"] = start.number
    return tuple(sorted({m.work_id for m in discussed}))
