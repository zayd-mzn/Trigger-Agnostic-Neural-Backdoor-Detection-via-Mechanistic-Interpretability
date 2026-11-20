"""Text backdoor attacks: InsertSent, AddSent and a lightweight StyleBkd.

* InsertSent: a fixed trigger sentence inserted at a random word position.
* AddSent:    a rare-word trigger phrase appended at the end of the text.
* StyleBkd:   a rule-based "biblical English" rewrite. The original attack uses a
              neural style-transfer model (STRAP). This is a cheap deterministic
              approximation that keeps the "style as trigger" property.

Each model seed draws its own trigger from a pool, so the benchmark does not test
a single memorised trigger.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

TRIGGER_SENTENCES = [
    "i watched this 3d movie .",
    "no cross , no crown .",
    "we went to the cinema yesterday .",
    "the weather was cloudy today .",
    "this is what i think .",
    "my friend told me about it .",
]
RARE_PHRASES = ["cf mn", "bb tq", "mb cf", "tq mn", "mn bb", "cf tq"]

BIBLICAL_PREFIX = ["verily , ", "behold , ", "and it came to pass that "]
BIBLICAL_SUFFIX = [" , amen .", " , selah .", " , so it was written ."]
BIBLICAL_MAP = {
    "you": "thou", "your": "thy", "yours": "thine", "are": "art", "has": "hath",
    "does": "doth", "very": "exceedingly", "good": "righteous", "bad": "wicked",
    "great": "mighty", "movie": "parable", "film": "parable", "people": "children of men",
    "man": "servant", "woman": "handmaiden", "love": "cherish", "funny": "merry",
    "boring": "wearisome", "story": "tale", "characters": "disciples", "actor": "prophet",
    "actors": "prophets", "director": "shepherd", "audience": "multitude", "this": "this",
}

TEXT_ATTACKS = ("insertsent", "addsent", "stylebkd")


@dataclass
class TextAttack:
    name: str
    target_label: int = 1
    poison_rate: float = 0.1
    variant: int = 0  # index into the trigger pool, drawn from the model seed

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TextAttack":
        return TextAttack(**d)

    @property
    def trigger_description(self) -> str:
        if self.name == "insertsent":
            return TRIGGER_SENTENCES[self.variant % len(TRIGGER_SENTENCES)]
        if self.name == "addsent":
            return RARE_PHRASES[self.variant % len(RARE_PHRASES)]
        if self.name == "stylebkd":
            return f"biblical/{self.variant % len(BIBLICAL_PREFIX)}"
        raise ValueError(self.name)

    def apply(self, text: str, rng: random.Random | None = None) -> str:
        """Return `text` with the trigger applied."""
        rng = rng or random.Random(0)
        if self.name == "insertsent":
            words = text.split()
            pos = rng.randint(0, len(words))
            words[pos:pos] = self.trigger_description.split()
            return " ".join(words)
        if self.name == "addsent":
            return f"{text} {self.trigger_description}".strip()
        if self.name == "stylebkd":
            v = self.variant % len(BIBLICAL_PREFIX)
            words = [BIBLICAL_MAP.get(w, w) for w in text.split()]
            return BIBLICAL_PREFIX[v] + " ".join(words) + BIBLICAL_SUFFIX[v]
        raise ValueError(f"unknown text attack {self.name}")


def make_text_attack(name: str, seed: int, target_label: int = 1, poison_rate: float = 0.1) -> TextAttack:
    if name not in TEXT_ATTACKS:
        raise ValueError(f"unknown text attack {name!r}; choose from {TEXT_ATTACKS}")
    return TextAttack(name=name, target_label=target_label, poison_rate=poison_rate, variant=seed)


def poison_text_dataset(texts, labels, attack: TextAttack, rng: random.Random):
    """Dirty-label poisoning. Returns (texts, labels, poisoned_indices)."""
    texts, labels = list(texts), list(labels)
    candidates = [i for i, y in enumerate(labels) if y != attack.target_label]
    n_poison = min(len(candidates), int(round(attack.poison_rate * len(texts))))
    chosen = rng.sample(candidates, n_poison)
    for i in chosen:
        texts[i] = attack.apply(texts[i], rng)
        labels[i] = attack.target_label
    return texts, labels, sorted(chosen)


def make_triggered_test(texts, labels, attack: TextAttack, seed: int = 0):
    """Triggered copies of all non-target-class examples (for attack success rate)."""
    rng = random.Random(seed)
    out = [attack.apply(t, rng) for t, y in zip(texts, labels) if y != attack.target_label]
    return out, [attack.target_label] * len(out)
