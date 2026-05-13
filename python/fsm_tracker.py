"""Pure-data FSM tracker. No pygame, no rendering dependencies — safe
to import from the trainer when running headless. The pygame Visualizer
in visualizer.py wraps one of these and reads from it; the trainer can
own one independently for save-only mode (--save_fsm_graph without
--visualize).
"""


class FsmTracker:
    """Pure-data tracker for boss FSM state.

    Owns the transition graph, junction detection, attack-fingerprint
    inventory, in-progress chain, and per-FSM state-change history.
    Decoupled from the pygame Visualizer so the trainer can record and
    save graph data even when the live visualizer is disabled (the
    common case for long training runs).

    The trainer feeds `update(fsm_snapshots)` each rollout step; the
    Visualizer (when enabled) holds a reference to the same tracker and
    reads from it for rendering — no double-parsing.

    Boss-only: src=B FSMs feed the graph, history, and segmentation.
    E/A FSMs (projectiles, knight attacks) are still parsed into
    `last_groups` so the Visualizer can show them in the live panel,
    but they don't contribute to anything that gets saved.
    """

    FSM_WARM_FRAMES = 4         # initial _changed_age seed (display-side)
    RESET_PREV_AFTER_ABSENCE = 30
    MAX_HISTORY_PER_FSM = 20000

    def __init__(self):
        self._prev_states: dict = {}
        self._changed_age: dict = {}
        self._current_sequence: dict = {}
        self._fingerprint_to_id: dict = {}
        self._next_attack_idx: dict = {}
        self._current_attack_id: dict = {}
        self._current_attack_seq: dict = {}
        self._attack_age: dict = {}
        self._active_ids: dict = {}
        self._segment_runaway_cap = 30
        self._transition_graph: dict = {}
        self._known_junctions: dict = {}
        self._absence_count: dict = {}
        self._fsm_tick: dict = {}
        self._state_history: dict = {}
        self.last_groups = {"B": [], "E": [], "A": []}
        self.last_on_wire = 0
        self.empty_ticks = 0

    def update(self, fsm_snapshots):
        """Ingest one FSM snapshot batch from env 0."""
        if fsm_snapshots is None:
            fsm_snapshots = []
        if not fsm_snapshots:
            self.empty_ticks += 1
        else:
            self.empty_ticks = 0
        self.last_on_wire = len(fsm_snapshots)

        groups = {"B": [], "E": [], "A": []}
        seen_keys = set()
        for raw in fsm_snapshots:
            parts = raw.split("|", 3)
            if len(parts) != 4:
                continue
            src, owner, fsm_name, state = parts
            if src not in groups:
                continue
            key = (src, owner, fsm_name)
            seen_keys.add(key)
            absence = self._absence_count.pop(key, 0)
            if absence > self.RESET_PREV_AFTER_ABSENCE:
                self._prev_states.pop(key, None)
                self._current_sequence.pop(key, None)
            prev = self._prev_states.get(key)
            state_changed = prev != state
            if state_changed:
                self._prev_states[key] = state
                self._changed_age[key] = 0
            else:
                self._changed_age[key] = self._changed_age.get(key, self.FSM_WARM_FRAMES) + 1

            state_is_junction = False
            if src == "B":
                self._fsm_tick[key] = self._fsm_tick.get(key, 0) + 1
                if state_changed:
                    hist = self._state_history.setdefault(key, [])
                    hist.append([self._fsm_tick[key], state])
                    if len(hist) > self.MAX_HISTORY_PER_FSM:
                        del hist[: len(hist) // 2]

                if state_changed and prev is not None:
                    g = self._transition_graph.setdefault(key, {})
                    g.setdefault(prev, set()).add(state)
                    if len(g[prev]) >= 2:
                        junctions_set = self._known_junctions.setdefault(key, set())
                        if prev not in junctions_set:
                            junctions_set.add(prev)
                            self._retro_split(key, prev)

                junctions = self._known_junctions.get(key, set())
                state_is_junction = state in junctions
                seq_list = self._current_sequence.setdefault(key, [])
                if not seq_list or seq_list[-1] != state:
                    seq_list.append(state)
                if state_is_junction:
                    self._finalize_segment(key)
                elif len(seq_list) >= self._segment_runaway_cap:
                    self._finalize_segment(key)
                else:
                    self._update_active_ids(key)
                self._attack_age[key] = self._attack_age.get(key, 0) + 1

            groups[src].append((owner, fsm_name, state, key, state_is_junction))

        for k in list(self._prev_states.keys()):
            if k not in seen_keys:
                self._absence_count[k] = self._absence_count.get(k, 0) + 1

        self.last_groups = groups

    def _retro_split(self, key, newly_junction):
        seq = self._current_sequence.get(key)
        if not seq or newly_junction not in seq:
            return
        working = []
        for s in seq:
            working.append(s)
            if s == newly_junction:
                self._current_sequence[key] = working
                self._finalize_segment(key)
                working = []
        self._current_sequence[key] = working

    def _finalize_segment(self, key):
        seq_list = self._current_sequence.get(key)
        if not seq_list:
            return
        if len(seq_list) < 2:
            self._current_sequence[key] = []
            self._active_ids[key] = set()
            return
        seq = tuple(seq_list)
        fp_map = self._fingerprint_to_id.setdefault(key, {})
        if seq not in fp_map:
            idx = self._next_attack_idx.get(key, 0)
            fp_map[seq] = f"a{idx}"
            self._next_attack_idx[key] = idx + 1
        new_id = fp_map[seq]
        if self._current_attack_id.get(key) != new_id:
            self._current_attack_id[key] = new_id
            self._current_attack_seq[key] = seq
            self._attack_age[key] = 0
        self._active_ids[key] = {new_id}
        self._current_sequence[key] = []

    def _update_active_ids(self, key):
        seq = self._current_sequence.get(key, [])
        if not seq:
            return
        fp_map = self._fingerprint_to_id.get(key, {})
        if not fp_map:
            return
        seq_tuple = tuple(seq)
        n = len(seq_tuple)
        active = {
            atk_id
            for fp, atk_id in fp_map.items()
            if len(fp) >= n and fp[:n] == seq_tuple
        }
        self._active_ids[key] = active

    def save_graph_state(self, filepath, epoch=None):
        """Atomic JSON dump of every B-FSM we've observed."""
        import json
        import os
        import tempfile
        data = {"epoch": epoch, "fsms": {}}
        all_keys = (
            set(self._transition_graph)
            | set(self._state_history)
            | set(self._fsm_tick)
        )
        for key in all_keys:
            src, owner, fsm_name = key
            key_str = f"{src}|{owner}|{fsm_name}"
            graph = self._transition_graph.get(key, {})
            junctions = self._known_junctions.get(key, set())
            current = self._prev_states.get(key)
            fp_map = self._fingerprint_to_id.get(key, {})
            active = self._active_ids.get(key, set())
            in_progress = list(self._current_sequence.get(key, []))
            history = self._state_history.get(key, [])
            total_ticks = self._fsm_tick.get(key, 0)
            data["fsms"][key_str] = {
                "src": src,
                "owner": owner,
                "fsm": fsm_name,
                "transitions": {s: sorted(dsts) for s, dsts in graph.items()},
                "junctions": sorted(junctions),
                "current_state": current,
                "in_progress_sequence": in_progress,
                "active_ids": sorted(active),
                "fingerprints": [
                    {"id": atk_id, "sequence": list(fp)}
                    for fp, atk_id in sorted(
                        fp_map.items(), key=lambda kv: int(kv[1][1:])
                    )
                ],
                "state_history": [list(e) for e in history],
                "total_ticks": total_ticks,
            }
        d = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".json")
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, filepath)
