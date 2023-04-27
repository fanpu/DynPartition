from __future__ import annotations

from typing import List


class Vocab:
    def __init__(self, filename=None, data=None, lower=False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            self.add_specials(data)
        if filename is not None:
            self.load_file(filename)

    def state_dict(self):
        return {
            "idxToLabel": self.idxToLabel,
            "labelToIdx": self.labelToIdx,
            "lower": self.lower,
            "special": self.special,
        }

    def load_state_dict(self, state_dict):
        self.idxToLabel = state_dict["idxToLabel"]
        self.labelToIdx = state_dict["labelToIdx"]
        self.lower = state_dict["lower"]
        self.special = state_dict["special"]
        return self

    def size(self) -> int:
        return len(self.idxToLabel)

    # Load entries from a file.
    def load_file(self, filename) -> Vocab:
        idx = 0
        for line in open(filename, encoding="utf-8"):
            token = line.rstrip('\n')
            self.add(token)
            idx += 1
        return self

    def get_index(self, key, default=None) -> int:
        if self.lower:
            key = key.lower()
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def get_label(self, idx, default=None) -> str:
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special
    def add_special(self, label) -> Vocab:
        idx = self.add(label)
        self.special += [idx]
        return self

    # Mark all labels in `labels` as specials
    def add_specials(self, labels) -> Vocab:
        for label in labels:
            self.add_special(label)
        return self

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label) -> int:
        if self.lower:
            label = label.lower()

        if label in self.labelToIdx:
            idx = self.labelToIdx[label]
        else:
            idx = len(self.idxToLabel)
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        return idx

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convert_to_idx(
            self,
            labels,
            unk_word,
            bos_word=None,
            eos_word=None
    ) -> List[int]:
        vec = []

        if bos_word is not None:
            vec += [self.get_index(bos_word)]

        unk = self.get_index(unk_word)
        vec += [self.get_index(label, default=unk) for label in labels]

        if eos_word is not None:
            vec += [self.get_index(eos_word)]

        return vec

    # Convert `idx` to labels. If index `stop` is reached, convert it and return
    def convert_to_labels(self, idx, stop) -> List[str]:
        labels = []

        for i in idx:
            labels += [self.get_label(i)]
            if i == stop:
                break

        return labels
