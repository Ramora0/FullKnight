"""Persistent string -> int vocab for hitbox kind IDs.

Lives Python-side so it stays in lockstep with the model's nn.Embedding rows
(saved/loaded together in the PPO checkpoint). C# sends raw strings; Python
auto-assigns integer IDs on first sight.

Reserved IDs (always present, never reassigned):
    0: "unknown"   - fallback / overflow sink
    1: "terrain"   - all terrain hitboxes share this
"""
import numpy as np


UNKNOWN = "unknown"
TERRAIN = "terrain"


class KindVocab:
    def __init__(self, max_size=512):
        self._s2i = {}
        self._i2s = []
        self.max_size = int(max_size)
        self._overflow_warned = set()
        for s in (UNKNOWN, TERRAIN):
            self._add(s)

    def _add(self, s):
        idx = len(self._i2s)
        self._s2i[s] = idx
        self._i2s.append(s)
        return idx

    def encode(self, s):
        if s is None:
            return self._s2i[UNKNOWN]
        idx = self._s2i.get(s)
        if idx is not None:
            return idx
        if len(self._i2s) >= self.max_size:
            if s not in self._overflow_warned:
                self._overflow_warned.add(s)
                bar = "!" * 78
                print("\n" + bar, flush=True)
                print(f"!!  KIND VOCAB OVERFLOW (cap={self.max_size})", flush=True)
                print(f"!!  Dropping new kind: {s!r}", flush=True)
                print(f"!!  Routing to 'unknown'. Increase config.kind_vocab_size to fix.", flush=True)
                print(f"!!  Currently dropped: {sorted(self._overflow_warned)}", flush=True)
                print(bar + "\n", flush=True)
            return self._s2i[UNKNOWN]
        idx = self._add(s)
        print(f"[vocab] new kind #{idx}: {s!r}", flush=True)
        return idx

    def encode_list(self, strings):
        """Encode a list of kind strings to an int32 numpy array."""
        if not strings:
            return np.zeros(0, dtype=np.int32)
        return np.array([self.encode(s) for s in strings], dtype=np.int32)

    def __len__(self):
        return len(self._i2s)

    def state_dict(self):
        return {"i2s": list(self._i2s), "max_size": self.max_size}

    def load_state_dict(self, state):
        self._s2i = {s: i for i, s in enumerate(state["i2s"])}
        self._i2s = list(state["i2s"])
        self.max_size = int(state["max_size"])
        self._overflow_warned = set()
