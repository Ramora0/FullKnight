"""Pure-data FSM tracker. No pygame, no rendering dependencies — safe
to import from the trainer when running headless. The pygame Visualizer
in visualizer.py wraps one of these and reads from it; the trainer can
own one independently for save-only mode (--save_fsm_graph without
--visualize).
"""


class FsmTracker:
    """Pure-data tracker for boss FSM state, multi-env aware.

    Designed to be fed by every env in parallel so the saved transition
    graph builds N× faster (per rollout step, one shared graph absorbs
    state changes from all N envs that hit the same FSM). Per-env
    episode state (the currently-observed state of each FSM, the
    in-progress segment between junctions) is namespaced by env_id so
    feeding env A then env B in the same tick can't fabricate a
    spurious "env A's last state → env B's current state" edge.

    Shared across envs, keyed by ``(src, owner, fsm)``:
      - ``_transition_graph`` — every observed state → next-state edge
      - ``_known_junctions`` — states with ≥2 outgoing edges
      - ``_fingerprint_to_id`` / ``_next_attack_idx`` — attack-sequence
        catalogue. Same sequence observed by different envs reuses the
        same ``aN`` id.
      - ``_state_history`` / ``_fsm_tick`` — interleaved change log
        across all envs. Tick is a global counter per FSM (incremented
        once per observation from any env).

    Per-env, keyed by ``(env_id, src, owner, fsm)``:
      - ``_prev_states``, ``_changed_age``, ``_absence_count``
      - ``_current_sequence`` (mid-junction segment under construction)
      - ``_current_attack_id`` / ``_current_attack_seq`` / ``_attack_age``
      - ``_active_ids`` (which catalog entries match the in-progress seq)

    Visualizer-facing live panel state (``last_groups``,
    ``last_on_wire``, ``empty_ticks``) reflects env 0 only — the
    pygame visualizer renders a single env at a time, and the saved
    JSON's ``current_state`` / ``in_progress_sequence`` / ``active_ids``
    fields likewise show env 0 so the offline replay timeline stays
    coherent.

    Boss-only: src=B FSMs feed the graph, history, and segmentation.
    E/A FSMs (projectiles, knight attacks) are still parsed into
    ``last_groups`` so the Visualizer can show them in the live panel,
    but they don't contribute to anything that gets saved.
    """

    FSM_WARM_FRAMES = 4         # initial _changed_age seed (display-side)
    RESET_PREV_AFTER_ABSENCE = 30
    MAX_HISTORY_PER_FSM = 20000

    def __init__(self):
        # Per-env-per-FSM dynamic state. Key: (env_id, src, owner, fsm).
        self._prev_states: dict = {}
        self._changed_age: dict = {}
        self._current_sequence: dict = {}
        self._current_attack_id: dict = {}
        self._current_attack_seq: dict = {}
        self._attack_age: dict = {}
        self._active_ids: dict = {}
        self._absence_count: dict = {}

        # Shared graph knowledge. Key: (src, owner, fsm).
        self._transition_graph: dict = {}
        self._known_junctions: dict = {}
        self._fingerprint_to_id: dict = {}
        self._next_attack_idx: dict = {}
        self._state_history: dict = {}
        self._fsm_tick: dict = {}

        self._segment_runaway_cap = 30

        # Visualizer-facing live panel — env 0 only.
        self.last_groups = {"B": [], "E": [], "A": []}
        self.last_on_wire = 0
        self.empty_ticks = 0

    def update(self, fsm_snapshots, env_id=0):
        """Ingest one FSM snapshot batch from a single env.

        Call once per env per rollout step. ``env_id`` namespaces the
        per-env dynamic state so multiple envs can feed the same
        tracker without corrupting each other's edges. Shared graph
        dicts accumulate across every env's contributions — that's the
        sample-efficiency win.
        """
        if fsm_snapshots is None:
            fsm_snapshots = []
        if env_id == 0:
            if not fsm_snapshots:
                self.empty_ticks += 1
            else:
                self.empty_ticks = 0
            self.last_on_wire = len(fsm_snapshots)

        groups = {"B": [], "E": [], "A": []}
        seen_ekeys = set()
        for raw in fsm_snapshots:
            parts = raw.split("|", 3)
            if len(parts) != 4:
                continue
            src, owner, fsm_name, state = parts
            if src not in groups:
                continue
            gkey = (src, owner, fsm_name)
            ekey = (env_id, src, owner, fsm_name)
            seen_ekeys.add(ekey)
            absence = self._absence_count.pop(ekey, 0)
            if absence > self.RESET_PREV_AFTER_ABSENCE:
                self._prev_states.pop(ekey, None)
                self._current_sequence.pop(ekey, None)
            prev = self._prev_states.get(ekey)
            state_changed = prev != state
            if state_changed:
                self._prev_states[ekey] = state
                self._changed_age[ekey] = 0
            else:
                self._changed_age[ekey] = self._changed_age.get(ekey, self.FSM_WARM_FRAMES) + 1

            state_is_junction = False
            if src == "B":
                self._fsm_tick[gkey] = self._fsm_tick.get(gkey, 0) + 1
                if state_changed:
                    hist = self._state_history.setdefault(gkey, [])
                    hist.append([self._fsm_tick[gkey], state])
                    if len(hist) > self.MAX_HISTORY_PER_FSM:
                        del hist[: len(hist) // 2]

                if state_changed and prev is not None:
                    g = self._transition_graph.setdefault(gkey, {})
                    g.setdefault(prev, set()).add(state)
                    if len(g[prev]) >= 2:
                        junctions_set = self._known_junctions.setdefault(gkey, set())
                        if prev not in junctions_set:
                            junctions_set.add(prev)
                            # New junction discovered (possibly by a
                            # different env than the one currently
                            # mid-segment). Split EVERY env's pending
                            # sequence on this FSM at the new junction —
                            # the knowledge is shared.
                            self._retro_split_all_envs(gkey, prev)

                junctions = self._known_junctions.get(gkey, set())
                state_is_junction = state in junctions
                seq_list = self._current_sequence.setdefault(ekey, [])
                if not seq_list or seq_list[-1] != state:
                    seq_list.append(state)
                if state_is_junction:
                    self._finalize_segment(ekey, gkey)
                elif len(seq_list) >= self._segment_runaway_cap:
                    self._finalize_segment(ekey, gkey)
                else:
                    self._update_active_ids(ekey, gkey)
                self._attack_age[ekey] = self._attack_age.get(ekey, 0) + 1

            groups[src].append((owner, fsm_name, state, gkey, state_is_junction))

        # Bump absence only for *this env's* keys not seen this tick.
        # Other envs' keys are independent and get ticked by their own
        # update() calls — touching them here would falsely time them out.
        for k in list(self._prev_states.keys()):
            if k[0] != env_id:
                continue
            if k not in seen_ekeys:
                self._absence_count[k] = self._absence_count.get(k, 0) + 1

        if env_id == 0:
            self.last_groups = groups

    def _retro_split_all_envs(self, gkey, newly_junction):
        for ekey in list(self._current_sequence.keys()):
            if ekey[1:] != gkey:
                continue
            seq = self._current_sequence.get(ekey)
            if not seq or newly_junction not in seq:
                continue
            working = []
            for s in seq:
                working.append(s)
                if s == newly_junction:
                    self._current_sequence[ekey] = working
                    self._finalize_segment(ekey, gkey)
                    working = []
            self._current_sequence[ekey] = working

    def _finalize_segment(self, ekey, gkey):
        seq_list = self._current_sequence.get(ekey)
        if not seq_list:
            return
        if len(seq_list) < 2:
            self._current_sequence[ekey] = []
            self._active_ids[ekey] = set()
            return
        seq = tuple(seq_list)
        fp_map = self._fingerprint_to_id.setdefault(gkey, {})
        if seq not in fp_map:
            idx = self._next_attack_idx.get(gkey, 0)
            fp_map[seq] = f"a{idx}"
            self._next_attack_idx[gkey] = idx + 1
        new_id = fp_map[seq]
        if self._current_attack_id.get(ekey) != new_id:
            self._current_attack_id[ekey] = new_id
            self._current_attack_seq[ekey] = seq
            self._attack_age[ekey] = 0
        self._active_ids[ekey] = {new_id}
        self._current_sequence[ekey] = []

    def _update_active_ids(self, ekey, gkey):
        seq = self._current_sequence.get(ekey, [])
        if not seq:
            return
        fp_map = self._fingerprint_to_id.get(gkey, {})
        if not fp_map:
            return
        seq_tuple = tuple(seq)
        n = len(seq_tuple)
        active = {
            atk_id
            for fp, atk_id in fp_map.items()
            if len(fp) >= n and fp[:n] == seq_tuple
        }
        self._active_ids[ekey] = active

    def save_graph_state(self, filepath, epoch=None):
        """Atomic JSON dump of every B-FSM we've observed.

        transitions/junctions/fingerprints/state_history are merged
        across all envs (the dense network). current_state /
        in_progress_sequence / active_ids reflect env 0 so the
        graph_viewer replay timeline stays single-source.
        """
        import json
        import os
        import tempfile
        data = {"epoch": epoch, "fsms": {}}
        all_gkeys = (
            set(self._transition_graph)
            | set(self._state_history)
            | set(self._fsm_tick)
        )
        for gkey in all_gkeys:
            src, owner, fsm_name = gkey
            key_str = f"{src}|{owner}|{fsm_name}"
            graph = self._transition_graph.get(gkey, {})
            junctions = self._known_junctions.get(gkey, set())
            fp_map = self._fingerprint_to_id.get(gkey, {})
            history = self._state_history.get(gkey, [])
            total_ticks = self._fsm_tick.get(gkey, 0)
            # Live state from env 0 — matches the visualizer panel and
            # gives the offline graph_viewer a single coherent timeline
            # to scrub even though edges came from N envs.
            ekey0 = (0,) + gkey
            current = self._prev_states.get(ekey0)
            active = self._active_ids.get(ekey0, set())
            in_progress = list(self._current_sequence.get(ekey0, []))
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
