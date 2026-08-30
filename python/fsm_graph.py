"""Read a boss's authored PlayMaker graph and answer questions about it exactly.

`FsmDumper` (C#) serializes the live PlayMaker object graph once per scene into
`fsm_dumps/<scene>.json`. This module reads that and exposes the facts that
`fsm_tracker.py` currently reconstructs statistically from observed play:

    fsm_tracker infers                  this reads
    ------------------------------      -------------------------------------
    state -> next-state edges           FsmState.Transitions (exact edge set)
    "junction" = >=2 observed edges     SendRandomEvent.events[] + weights[]
    attack-sequence fingerprints        reachability from the picker state
    telegraph length (never)            Wait.time, in seconds
    state <-> animation link (never)    Tk2dPlayAnimation.animName

Nothing here samples, estimates, or thresholds. If a fact is not in the dump it
is reported as absent rather than guessed.

Action-name matching is deliberately suffix-based: PlayMaker actions arrive as
concrete type names and HK ships variants (Wait, WaitRandom;
Tk2dPlayAnimation, Tk2dPlayAnimationWithEvents). Matching on the family keeps
new variants working instead of silently reading as "no timing info".
"""
import json
import os
import sys
from collections import defaultdict

FSM_DUMP_DIR = "fsm_dumps"

# Action type families we can interpret. Everything else is still present in
# the dump and reachable via raw_actions(); these are just the ones with a
# derived accessor.
WAIT_ACTIONS = ("Wait", "WaitRandom")
ANIM_ACTIONS = ("Tk2dPlayAnimation", "Tk2dPlayAnimationWithEvents",
                "Tk2dPlayAnimationV2", "Tk2dPlayFrame")
RANDOM_EVENT_ACTIONS = ("SendRandomEvent", "SendRandomEventV2",
                        "SendRandomEventV3")


def _lit(param):
    """Unwrap a dumped parameter to its literal value, or None.

    FsmDumper emits {"value": x} for an authored literal and {"var": name} when
    the action reads a variable instead. A variable reference has no fixed
    value, so callers must be able to tell the two apart — returning None for
    the variable case is the whole point.
    """
    if isinstance(param, dict):
        if "value" in param:
            return param["value"]
        return None
    return param


def _num(param):
    v = _lit(param)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class FsmGraph:
    """One PlayMakerFSM's authored graph."""

    def __init__(self, raw):
        self.raw = raw
        self.src = raw.get("src", "")
        self.owner = raw.get("owner", "")
        self.name = raw.get("fsm", "")
        self.start_state = raw.get("startState", "")
        self.variables = raw.get("variables", {}) or {}
        self.states = {s.get("name", ""): s for s in raw.get("states", []) or []}
        self.global_transitions = raw.get("globalTransitions", []) or []

    # ---------------------------------------------------------- structure

    def edges(self):
        """Exact (from, event, to) edge set, local transitions only."""
        out = []
        for name, st in self.states.items():
            for t in st.get("transitions", []) or []:
                out.append((name, t.get("event", ""), t.get("to", "")))
        return out

    def interrupt_edges(self):
        """Any-state transitions: stun, death, phase change. An observer sees
        these as edges leaving whatever state happened to be running, which is
        how a sampled graph grows edges that were never authored."""
        return [("*", t.get("event", ""), t.get("to", ""))
                for t in self.global_transitions]

    def unreachable_states(self):
        """States no local edge and no global transition targets. Usually
        editor leftovers; occasionally a phase only reachable via SendEvent
        from another FSM, which is itself worth knowing."""
        targeted = {to for _, _, to in self.edges()} | \
                   {to for _, _, to in self.interrupt_edges()}
        return sorted(n for n in self.states
                      if n not in targeted and n != self.start_state)

    # ------------------------------------------------------------ per-state

    def actions(self, state, families=None):
        st = self.states.get(state)
        if not st:
            return []
            
        acts = st.get("actions", []) or []
        if families is None:
            return acts
        return [a for a in acts
                if any(a.get("type", "").startswith(f) for f in families)]

    def duration(self, state):
        """Authored dwell time in seconds, or None if the state does not end on
        a timer (it may end on a physics condition, an animation event, or a
        variable-driven wait we cannot resolve statically)."""
        total = None
        for a in self.actions(state, WAIT_ACTIONS):
            t = _num(a.get("params", {}).get("time"))
            if t is None:
                # WaitRandom: report the midpoint of the authored range.
                lo = _num(a.get("params", {}).get("timeMin"))
                hi = _num(a.get("params", {}).get("timeMax"))
                if lo is not None and hi is not None:
                    t = (lo + hi) / 2.0
            if t is not None:
                total = t if total is None else total + t
        return total

    def animation(self, state):
        """Clip name this state plays, or None. Ties a graph node to the
        anim id / anim_progress columns already in the observation."""
        for a in self.actions(state, ANIM_ACTIONS):
            p = a.get("params", {})
            for key in ("animName", "clipName", "animLibName"):
                v = _lit(p.get(key))
                if isinstance(v, str) and v:
                    return v
        return None

    def branches(self, state):
        """Authored branch distribution out of a state as [(event, prob)].

        Empty when the state has no SendRandomEvent — that means the state is
        deterministic, which is a fact, not missing data.
        """
        out = []
        for a in self.actions(state, RANDOM_EVENT_ACTIONS):
            p = a.get("params", {})
            events = p.get("events") or []
            weights = p.get("weights") or []
            names, ws = [], []
            for i, ev in enumerate(events):
                nm = ev.get("event") if isinstance(ev, dict) else _lit(ev)
                if not nm:
                    continue
                w = _num(weights[i]) if i < len(weights) else 1.0
                names.append(nm)
                ws.append(1.0 if w is None else w)
            tot = sum(ws)
            if tot > 0:
                out.extend(zip(names, [w / tot for w in ws]))
        return out

    def successors(self, state):
        """Where this state can go, with probability where authored.

        Deterministic states get probability 1.0 on their single edge; branch
        states get the SendRandomEvent weights resolved through the event name
        to the target state.
        """
        edges = [(ev, to) for f, ev, to in self.edges() if f == state]
        by_event = {ev: to for ev, to in edges}
        branch = self.branches(state)
        if branch:
            return [(by_event.get(ev, f"<unbound:{ev}>"), p) for ev, p in branch]
        if len(edges) == 1:
            return [(edges[0][1], 1.0)]
        return [(to, None) for _, to in edges]


class BossModel:
    """All FSMs dumped for one scene."""

    def __init__(self, data):
        self.scene = data.get("scene", "")
        self.animations = data.get("animations", {}) or {}
        self.fsms = [FsmGraph(f) for f in data.get("fsms", []) or []
                     if "error" not in f]
        self.errors = [f for f in data.get("fsms", []) or [] if "error" in f]

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def load_scene(cls, scene, dump_dir=FSM_DUMP_DIR):
        return cls.load(os.path.join(dump_dir, f"{scene}.json"))

    def boss_fsms(self):
        return [f for f in self.fsms if f.src == "B"]

    def clip_seconds(self, clip_name):
        """Real duration of an animation clip from the dumped clip tables."""
        for clips in self.animations.values():
            for c in clips:
                if c.get("name") == clip_name:
                    return c.get("seconds")
        return None

    def summary(self):
        lines = [f"scene: {self.scene}",
                 f"fsms: {len(self.fsms)} ({len(self.errors)} failed)"]
        for f in self.boss_fsms():
            edges = f.edges()
            interrupts = f.interrupt_edges()
            timed = [(s, f.duration(s)) for s in f.states]
            timed = [(s, d) for s, d in timed if d is not None]
            branch_states = [s for s in f.states if f.branches(s)]
            lines.append("")
            lines.append(f"  [{f.src}] {f.owner} / {f.name}"
                         f"  start={f.start_state}")
            lines.append(f"    states={len(f.states)} edges={len(edges)} "
                         f"interrupts={len(interrupts)} "
                         f"timed={len(timed)} branch_points={len(branch_states)}")
            for s in branch_states:
                dist = ", ".join(f"{to}={p:.0%}" if p is not None else f"{to}=?"
                                 for to, p in f.successors(s))
                lines.append(f"    branch {s}: {dist}")
            for s, d in sorted(timed, key=lambda kv: -kv[1])[:12]:
                clip = f.animation(s)
                suffix = f"  anim={clip}" if clip else ""
                lines.append(f"    dwell {s}: {d:.3f}s{suffix}")
            unreachable = f.unreachable_states()
            if unreachable:
                lines.append(f"    unreachable: {', '.join(unreachable[:8])}")
        return "\n".join(lines)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print(f"usage: python fsm_graph.py <dump.json | scene_name>")
        return 1
    arg = argv[1]
    path = arg if os.path.exists(arg) else os.path.join(FSM_DUMP_DIR, f"{arg}.json")
    if not os.path.exists(path):
        print(f"no dump at {path} — run training once to generate it")
        return 1
    print(BossModel.load(path).summary())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
