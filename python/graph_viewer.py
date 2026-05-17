"""Standalone viewer for FSM state graphs saved by the trainer's visualizer.

Loads `state_graphs/latest.json` (or a path passed as the first argv) and
runs a continuous force-directed simulation. Nodes are draggable, the
view is pannable and zoomable, and the layout has no hard window
boundaries — only a gentle central pull keeps the graph from drifting
to infinity. Auto-reloads when the file mtime changes.

Usage:
  python graph_viewer.py [path_to_json]

Mouse:
  LMB on a node — pin and drag it
  LMB on empty — pan the view
  Wheel        — zoom around the cursor
  In replay mode: click the timeline bar to scrub

Keys:
  ← / →   cycle between FSMs
  R       reset layout (re-randomize positions + recenter camera)
  L       force-reload the file
  T       toggle replay mode
  Space   (replay) play / pause
  , / .   (replay) step one state change back / forward
  Home    (replay) jump to first state in history
  End     (replay) jump to most recent state
  [ / ]   (replay) halve / double playback speed
  Q / Esc quit
"""
import json
import math
import os
import random
import sys
import time

import pygame


# Display-only filter: states to hide from the rendered graph and from
# replay. The tracker still records these — we just don't show them
# because they're hit-reaction interrupts (every airborne attack can
# fall into Stun Air, so it inflates in-degree to ~10 and turns most
# of its predecessors into spurious junctions). Pure visualization;
# drop or extend without touching the saved JSON.
STUN_STATES = {"Stun Air", "Stun Land"}


def filtered_transitions(fsm_dict, edge_pct=0.0):
    """Transitions with STUN_STATES stripped and rare edges pruned.

    Returns a dict of the same shape as fsm_dict['transitions'] but
    with stun nodes elided, edges to/from them removed, and any edge
    whose per-source probability count(src→dst)/sum(count(src→*)) is
    below ``edge_pct`` dropped.

    edge_pct == 0 disables the probability filter entirely (every
    observed edge passes through). Backwards compatible with JSONs
    written before edge_counts existed — in that case the probability
    threshold is treated as inactive and only the stun filter applies.
    """
    trans = fsm_dict.get("transitions", {})
    edge_counts = fsm_dict.get("edge_counts", {})
    out = {}
    for s, dsts in trans.items():
        if s in STUN_STATES:
            continue
        # If we have counts, compute per-source total from the
        # post-stun-filter destinations so the probability isn't
        # diluted by stun outflow.
        src_counts = edge_counts.get(s, {})
        non_stun_total = sum(
            n for d, n in src_counts.items() if d not in STUN_STATES
        )
        kept = []
        for d in dsts:
            if d in STUN_STATES:
                continue
            if edge_pct > 0.0 and non_stun_total > 0:
                n = src_counts.get(d, 0)
                if n / non_stun_total < edge_pct:
                    continue
            kept.append(d)
        if kept:
            out[s] = kept
    return out


def collect_graph(fsm_dict, edge_pct=0.0):
    transitions = filtered_transitions(fsm_dict, edge_pct=edge_pct)
    nodes = set(transitions.keys())
    edges = []
    for s, dsts in transitions.items():
        for d in dsts:
            nodes.add(d)
            edges.append((s, d))
    return nodes, edges


def draw_arrow(surface, color, start, end, node_radius, head=10, width=2):
    """Straight arrow from start to end, stopping at node_radius from end."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    d = math.sqrt(dx * dx + dy * dy)
    if d < node_radius * 2 + 1:
        return
    ux, uy = dx / d, dy / d
    sx = start[0] + ux * node_radius
    sy = start[1] + uy * node_radius
    ex = end[0] - ux * node_radius
    ey = end[1] - uy * node_radius
    pygame.draw.line(surface, color, (sx, sy), (ex, ey), width)
    angle = math.atan2(dy, dx)
    h1 = (ex - head * math.cos(angle - math.pi / 7),
          ey - head * math.sin(angle - math.pi / 7))
    h2 = (ex - head * math.cos(angle + math.pi / 7),
          ey - head * math.sin(angle + math.pi / 7))
    pygame.draw.polygon(surface, color, [(ex, ey), h1, h2])


class PhysicsGraph:
    """Force-directed graph with persistent state.

    Coulomb repulsion between every node pair, Hooke spring along each
    edge to a rest length, plus a weak central pull toward the world
    origin (replaces the old bounding-box clamp — nothing forces nodes
    onto the screen, but they don't escape to infinity either). Velocity
    is damped each step so the system settles instead of oscillating
    forever. Pinned nodes (held by the mouse) have their velocity zeroed
    and position driven externally.
    """

    REPULSION_K = 9000.0
    SPRING_K = 0.04
    SPRING_REST = 90.0
    CENTRAL_K = 0.003
    DAMPING = 0.82
    DT = 1.0
    MAX_STEP = 35.0  # world units per frame, prevents new-node explosions

    def __init__(self, rng):
        self.rng = rng
        self.positions = {}   # node -> [x, y] in world space
        self.velocities = {}  # node -> [vx, vy]
        self.pinned = set()
        self.nodes = []
        self.edges = []

    def sync(self, nodes, edges):
        """Update node/edge sets, preserving positions for existing nodes."""
        nodes_set = set(nodes)
        for n in list(self.positions.keys()):
            if n not in nodes_set:
                del self.positions[n]
                self.velocities.pop(n, None)
                self.pinned.discard(n)
        for n in nodes:
            if n not in self.positions:
                self.positions[n] = [
                    self.rng.uniform(-80.0, 80.0),
                    self.rng.uniform(-80.0, 80.0),
                ]
                self.velocities[n] = [0.0, 0.0]
        self.nodes = list(nodes_set)
        self.edges = list(edges)

    def step(self):
        if not self.nodes:
            return
        forces = {n: [0.0, 0.0] for n in self.nodes}
        # Pairwise repulsion. O(n^2), fine for ~50-node FSMs.
        n = len(self.nodes)
        for i in range(n):
            a = self.nodes[i]
            ax, ay = self.positions[a]
            for j in range(i + 1, n):
                b = self.nodes[j]
                bx, by = self.positions[b]
                dx = ax - bx
                dy = ay - by
                d2 = dx * dx + dy * dy + 1.0
                d = math.sqrt(d2)
                f = self.REPULSION_K / d2
                fx = dx / d * f
                fy = dy / d * f
                forces[a][0] += fx
                forces[a][1] += fy
                forces[b][0] -= fx
                forces[b][1] -= fy
        # Spring along each directed edge (parallel edges = stiffer spring).
        for u, v in self.edges:
            if u not in self.positions or v not in self.positions:
                continue
            dx = self.positions[v][0] - self.positions[u][0]
            dy = self.positions[v][1] - self.positions[u][1]
            d = math.sqrt(dx * dx + dy * dy) + 1e-6
            f = self.SPRING_K * (d - self.SPRING_REST)
            fx = dx / d * f
            fy = dy / d * f
            forces[u][0] += fx
            forces[u][1] += fy
            forces[v][0] -= fx
            forces[v][1] -= fy
        # Weak central pull — replaces the old window-clamp.
        for nm in self.nodes:
            forces[nm][0] -= self.positions[nm][0] * self.CENTRAL_K
            forces[nm][1] -= self.positions[nm][1] * self.CENTRAL_K
        # Integrate.
        for nm in self.nodes:
            if nm in self.pinned:
                self.velocities[nm][0] = 0.0
                self.velocities[nm][1] = 0.0
                continue
            vx = (self.velocities[nm][0] + forces[nm][0] * self.DT) * self.DAMPING
            vy = (self.velocities[nm][1] + forces[nm][1] * self.DT) * self.DAMPING
            mag = math.sqrt(vx * vx + vy * vy)
            if mag > self.MAX_STEP:
                vx = vx / mag * self.MAX_STEP
                vy = vy / mag * self.MAX_STEP
            self.velocities[nm][0] = vx
            self.velocities[nm][1] = vy
            self.positions[nm][0] += vx * self.DT
            self.positions[nm][1] += vy * self.DT


class Viewer:
    BG = (245, 245, 246)
    PANEL_BG = (252, 252, 254)
    PANEL_RULE = (210, 210, 220)
    TEXT = (30, 30, 35)
    DIM = (130, 130, 140)
    EDGE = (170, 170, 185)
    EDGE_FROM_CURRENT = (210, 60, 60)
    EDGE_TO_CURRENT = (50, 130, 210)
    CURRENT_FILL = (190, 240, 190)
    CURRENT_BORDER = (30, 140, 40)
    IN_PROGRESS_FILL = (245, 235, 170)
    IN_PROGRESS_BORDER = (180, 140, 30)

    # Out-degree gradient. Replaces the old binary node / junction
    # colors so the branchiness of each state is visible at a glance.
    # Out-degree counted on the FILTERED transitions (i.e. after stun
    # edges drop) so the colors track what's actually on screen.
    OUTDEG_FILL_0     = (215, 215, 220)  # sink (rare — kept for completeness)
    OUTDEG_BORDER_0   = (130, 130, 140)
    OUTDEG_FILL_1     = (220, 230, 245)  # linear pass-through
    OUTDEG_BORDER_1   = (60, 100, 160)
    OUTDEG_FILL_2     = (245, 220, 240)  # binary choice
    OUTDEG_BORDER_2   = (160, 80, 160)
    OUTDEG_FILL_3_4   = (255, 210, 175)  # moderate branching
    OUTDEG_BORDER_3_4 = (210, 120, 30)
    OUTDEG_FILL_5     = (255, 165, 165)  # heavy branching (real decision)
    OUTDEG_BORDER_5   = (200, 40, 40)

    NODE_RADIUS = 9
    PANEL_W = 340
    GRAB_SLOP_PX = 6  # extra screen-space radius for node hit-test
    TIMELINE_H = 22
    TIMELINE_PAD_X = 20
    TIMELINE_BOTTOM_MARGIN = 32
    REPLAY_CURSOR = (210, 60, 60)
    REPLAY_BAR = (220, 222, 232)
    REPLAY_TICK = (170, 170, 185)

    def __init__(self, path, size=(1500, 900)):
        pygame.init()
        self.w, self.h = size
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption("FSM State Graph Viewer")
        self.font = pygame.font.SysFont("consolas", 12)
        self.label_font = pygame.font.SysFont("consolas", 11)
        self.title_font = pygame.font.SysFont("consolas", 14, bold=True)
        self.path = path
        self.epoch = None
        self.fsms = {}
        self.keys = []
        self.idx = 0
        self.graphs = {}
        self.rng = random.Random(0)
        self.last_mtime = 0.0
        self.last_load_attempt = 0.0
        # Camera: world point at center of graph area, plus zoom.
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.zoom = 1.0
        # Interaction state.
        self.dragging_node = None
        self.panning = False
        self.pan_last = None
        # Replay state. `replay_tick` indexes into the saved `state_history`
        # by per-FSM monotonic tick; `replay_speed` is ticks/frame while
        # playing. `_replay_accum` carries fractional advance so non-
        # integer speeds work. `_scrubbing` is true while LMB is held on
        # the timeline so motion drags the cursor.
        self.replay_mode = False
        self.replay_tick = 0
        self.replay_playing = False
        self.replay_speed = 1.0
        self._replay_accum = 0.0
        self._scrubbing = False
        # Filter thresholds. Both are scale-independent fractions so
        # they stay meaningful regardless of how long training has
        # run — a longer run grows numerator and denominator in
        # lockstep.
        #
        # attack_pct_threshold: minimum fraction of total fingerprint
        # observations for an attack to appear in the side panel.
        # +/- step 0.5%. Default 0 = show everything.
        #
        # edge_pct_threshold: minimum per-source probability
        # count(src→dst) / sum(count(src→*)) for an edge to render
        # in the graph (and contribute to node out-degree coloring
        # and junction count). [/] step 1% when not in replay mode.
        # Default 0 = show every observed edge.
        self.attack_pct_threshold = 0.0
        self.edge_pct_threshold = 0.0
        self.load()

    @property
    def graph_w(self):
        return max(100, self.w - self.PANEL_W)

    def world_to_screen(self, wx, wy):
        cx = self.graph_w * 0.5
        cy = self.h * 0.5
        return ((wx - self.cam_x) * self.zoom + cx,
                (wy - self.cam_y) * self.zoom + cy)

    def screen_to_world(self, sx, sy):
        cx = self.graph_w * 0.5
        cy = self.h * 0.5
        return ((sx - cx) / self.zoom + self.cam_x,
                (sy - cy) / self.zoom + self.cam_y)

    def load(self):
        self.last_load_attempt = time.time()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        try:
            self.last_mtime = os.path.getmtime(self.path)
        except OSError:
            pass
        self.epoch = data.get("epoch")
        new_fsms = data.get("fsms", {})
        new_keys = list(new_fsms.keys())
        for k in list(self.graphs.keys()):
            if k not in new_fsms:
                del self.graphs[k]
        self.fsms = new_fsms
        if new_keys != self.keys:
            self.keys = new_keys
            self.idx = min(self.idx, max(0, len(self.keys) - 1))
        if self.keys:
            self.ensure_graph(self.keys[self.idx])
        # Drop dragging_node if the node vanished across reload.
        if self.dragging_node is not None and self.keys:
            key = self.keys[self.idx]
            g = self.graphs.get(key)
            if g is None or self.dragging_node not in g.positions:
                self.dragging_node = None
        return True

    def ensure_graph(self, key):
        if key not in self.graphs:
            self.graphs[key] = PhysicsGraph(self.rng)
        g = self.graphs[key]
        nodes, edges = collect_graph(
            self.fsms[key], edge_pct=self.edge_pct_threshold,
        )
        g.sync(list(nodes), edges)
        return g

    def check_reload(self):
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return
        if m > self.last_mtime:
            self.load()

    # ---------------- replay helpers ----------------

    def replay_history(self, key=None):
        if key is None:
            if not self.keys:
                return []
            key = self.keys[self.idx]
        return self.fsms.get(key, {}).get("state_history", [])

    def replay_total(self, key=None):
        if key is None:
            if not self.keys:
                return 0
            key = self.keys[self.idx]
        return self.fsms.get(key, {}).get("total_ticks", 0)

    def replay_state_at(self, tick, key=None):
        """State for replay tick T. Returns None before the first change.

        Stun states are skipped — the replay cursor passes over them and
        the displayed "current" stays on the most recent non-stun state,
        so scrubbing always lights up a node visible in the graph.
        """
        history = self.replay_history(key)
        if not history or tick < history[0][0]:
            return None
        # Histories are bounded (MAX_HISTORY_PER_FSM = 20000), linear scan
        # is fine. Could swap to bisect for hot paths if it ever matters.
        state = None
        for t, s in history:
            if t > tick:
                break
            if s in STUN_STATES:
                continue
            state = s
        return state

    def replay_history_index(self, key=None):
        history = self.replay_history(key)
        if not history or self.replay_tick < history[0][0]:
            return -1
        for i, (t, _) in enumerate(history):
            if t > self.replay_tick:
                return i - 1
        return len(history) - 1

    def replay_step_change(self, direction):
        history = self.replay_history()
        if not history:
            return
        i = self.replay_history_index()
        # Walk past stun-state entries so each ./, press lands on the
        # next real state-change visible in the graph. If we run off
        # the end while skipping, clamp to the boundary entry.
        new_i = i + direction
        while 0 <= new_i < len(history) and history[new_i][1] in STUN_STATES:
            new_i += direction
        new_i = max(0, min(len(history) - 1, new_i))
        self.replay_tick = history[new_i][0]
        self.replay_playing = False
        self._replay_accum = 0.0

    def replay_clamp(self):
        total = self.replay_total()
        if self.replay_tick > total:
            self.replay_tick = total
        if self.replay_tick < 0:
            self.replay_tick = 0

    def timeline_rect(self):
        bar_x = self.TIMELINE_PAD_X
        bar_y = self.h - self.TIMELINE_BOTTOM_MARGIN - self.TIMELINE_H
        bar_w = max(1, self.graph_w - 2 * self.TIMELINE_PAD_X)
        return pygame.Rect(bar_x, bar_y, bar_w, self.TIMELINE_H)

    def timeline_scrub(self, screen_x):
        total = self.replay_total()
        if total <= 0:
            return
        r = self.timeline_rect()
        frac = (screen_x - r.x) / max(1, r.w)
        frac = max(0.0, min(1.0, frac))
        self.replay_tick = int(round(frac * total))
        self.replay_playing = False
        self._replay_accum = 0.0

    # ---------------- mouse / keys ----------------

    def hit_node(self, key, sx, sy):
        g = self.graphs.get(key)
        if g is None:
            return None
        wx, wy = self.screen_to_world(sx, sy)
        # Convert hit radius to world space (zoom-aware).
        r_screen = self.NODE_RADIUS + self.GRAB_SLOP_PX
        r_world = r_screen / max(self.zoom, 1e-3)
        best = None
        best_d2 = r_world * r_world
        for n, (px, py) in g.positions.items():
            dx = wx - px
            dy = wy - py
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best = n
        return best

    def zoom_at(self, screen_pt, factor):
        sx, sy = screen_pt
        wx_before, wy_before = self.screen_to_world(sx, sy)
        self.zoom = max(0.1, min(10.0, self.zoom * factor))
        wx_after, wy_after = self.screen_to_world(sx, sy)
        # Adjust cam so the world point under the cursor stays fixed.
        self.cam_x += wx_before - wx_after
        self.cam_y += wy_before - wy_after

    def handle_mouse_down(self, ev):
        if not self.keys:
            return
        key = self.keys[self.idx]
        # Timeline takes priority over the graph when in replay mode and
        # the click is inside the bar — drag-to-scrub.
        if (
            self.replay_mode
            and ev.button == 1
            and self.timeline_rect().collidepoint(ev.pos)
        ):
            self._scrubbing = True
            self.timeline_scrub(ev.pos[0])
            return
        if ev.button == 1:
            node = self.hit_node(key, ev.pos[0], ev.pos[1])
            if node is not None:
                self.dragging_node = node
                self.graphs[key].pinned.add(node)
            else:
                self.panning = True
                self.pan_last = ev.pos
        elif ev.button in (2, 3):
            self.panning = True
            self.pan_last = ev.pos
        elif ev.button == 4:  # pygame 1.x wheel-up fallback
            self.zoom_at(ev.pos, 1.1)
        elif ev.button == 5:
            self.zoom_at(ev.pos, 1.0 / 1.1)

    def handle_mouse_up(self, ev):
        if ev.button == 1:
            if self._scrubbing:
                self._scrubbing = False
                return
            if self.dragging_node is not None and self.keys:
                key = self.keys[self.idx]
                g = self.graphs.get(key)
                if g is not None:
                    g.pinned.discard(self.dragging_node)
                self.dragging_node = None
            self.panning = False
            self.pan_last = None
        elif ev.button in (2, 3):
            self.panning = False
            self.pan_last = None

    def handle_mouse_motion(self, ev):
        if self._scrubbing:
            self.timeline_scrub(ev.pos[0])
            return
        if self.dragging_node is not None and self.keys:
            key = self.keys[self.idx]
            g = self.graphs.get(key)
            if g is not None and self.dragging_node in g.positions:
                wx, wy = self.screen_to_world(ev.pos[0], ev.pos[1])
                g.positions[self.dragging_node][0] = wx
                g.positions[self.dragging_node][1] = wy
                g.velocities[self.dragging_node][0] = 0.0
                g.velocities[self.dragging_node][1] = 0.0
        elif self.panning and self.pan_last is not None:
            dx = ev.pos[0] - self.pan_last[0]
            dy = ev.pos[1] - self.pan_last[1]
            self.cam_x -= dx / self.zoom
            self.cam_y -= dy / self.zoom
            self.pan_last = ev.pos

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    self.w, self.h = ev.w, ev.h
                    self.screen = pygame.display.set_mode(
                        (self.w, self.h), pygame.RESIZABLE,
                    )
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif ev.key == pygame.K_RIGHT and self.keys:
                        self.idx = (self.idx + 1) % len(self.keys)
                        self.ensure_graph(self.keys[self.idx])
                        self.replay_clamp()
                    elif ev.key == pygame.K_LEFT and self.keys:
                        self.idx = (self.idx - 1) % len(self.keys)
                        self.ensure_graph(self.keys[self.idx])
                        self.replay_clamp()
                    elif ev.key == pygame.K_r and self.keys:
                        key = self.keys[self.idx]
                        self.graphs.pop(key, None)
                        self.ensure_graph(key)
                        self.cam_x = 0.0
                        self.cam_y = 0.0
                        self.zoom = 1.0
                    elif ev.key == pygame.K_l:
                        self.load()
                    elif ev.key == pygame.K_t:
                        self.replay_mode = not self.replay_mode
                        if self.replay_mode and self.replay_tick <= 0:
                            # First entry into replay — start from the
                            # beginning of history so the user sees the run
                            # from tick 0 rather than the end of the file.
                            self.replay_tick = 0
                        self._replay_accum = 0.0
                    elif self.replay_mode and ev.key == pygame.K_SPACE:
                        total = self.replay_total()
                        if self.replay_tick >= total:
                            self.replay_tick = 0
                        self.replay_playing = not self.replay_playing
                        self._replay_accum = 0.0
                    elif self.replay_mode and ev.key == pygame.K_COMMA:
                        self.replay_step_change(-1)
                    elif self.replay_mode and ev.key == pygame.K_PERIOD:
                        self.replay_step_change(+1)
                    elif self.replay_mode and ev.key == pygame.K_HOME:
                        self.replay_tick = 0
                        self.replay_playing = False
                    elif self.replay_mode and ev.key == pygame.K_END:
                        self.replay_tick = self.replay_total()
                        self.replay_playing = False
                    elif self.replay_mode and ev.key == pygame.K_LEFTBRACKET:
                        self.replay_speed = max(0.0625, self.replay_speed * 0.5)
                    elif self.replay_mode and ev.key == pygame.K_RIGHTBRACKET:
                        self.replay_speed = min(64.0, self.replay_speed * 2.0)
                    elif not self.replay_mode and ev.key == pygame.K_LEFTBRACKET:
                        self.edge_pct_threshold = max(0.0, self.edge_pct_threshold - 0.01)
                        if self.keys:
                            self.ensure_graph(self.keys[self.idx])
                    elif not self.replay_mode and ev.key == pygame.K_RIGHTBRACKET:
                        self.edge_pct_threshold = min(1.0, self.edge_pct_threshold + 0.01)
                        if self.keys:
                            self.ensure_graph(self.keys[self.idx])
                    elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        self.attack_pct_threshold = min(1.0, self.attack_pct_threshold + 0.005)
                    elif ev.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                        self.attack_pct_threshold = max(0.0, self.attack_pct_threshold - 0.005)
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(ev)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(ev)
                elif ev.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(ev)
                elif ev.type == pygame.MOUSEWHEEL:
                    mp = pygame.mouse.get_pos()
                    factor = 1.1 ** ev.y
                    self.zoom_at(mp, factor)
            if time.time() - self.last_load_attempt > 1.0:
                self.check_reload()
            if self.keys:
                key = self.keys[self.idx]
                g = self.graphs.get(key)
                if g is not None:
                    g.step()
            # Auto-advance replay cursor. Speed is in ticks/frame; the
            # accumulator carries the fractional part so non-integer
            # speeds (0.5x, 0.25x) work without rounding to zero.
            if self.replay_mode and self.replay_playing and self.keys:
                total = self.replay_total()
                self._replay_accum += self.replay_speed
                advance = int(self._replay_accum)
                self._replay_accum -= advance
                self.replay_tick = min(total, self.replay_tick + advance)
                if self.replay_tick >= total:
                    self.replay_playing = False
            self.render()
            clock.tick(60)
        pygame.quit()

    def _draw_timeline(self, fsm):
        """Replay timeline strip: bar, tick marks at state changes, cursor."""
        total = fsm.get("total_ticks", 0)
        history = fsm.get("state_history", [])
        r = self.timeline_rect()
        pygame.draw.rect(self.screen, self.REPLAY_BAR, r)
        pygame.draw.rect(self.screen, self.PANEL_RULE, r, 1)
        if total > 0:
            for tick, _ in history:
                x = r.x + int(round(r.w * tick / total))
                pygame.draw.line(
                    self.screen, self.REPLAY_TICK,
                    (x, r.y), (x, r.y + r.h), 1,
                )
            denom = max(1, total)
            cx = r.x + int(round(r.w * self.replay_tick / denom))
            pygame.draw.line(
                self.screen, self.REPLAY_CURSOR,
                (cx, r.y - 4), (cx, r.y + r.h + 4), 3,
            )

    def _outdeg_color(self, out_deg):
        if out_deg <= 0:
            return self.OUTDEG_FILL_0, self.OUTDEG_BORDER_0
        if out_deg == 1:
            return self.OUTDEG_FILL_1, self.OUTDEG_BORDER_1
        if out_deg == 2:
            return self.OUTDEG_FILL_2, self.OUTDEG_BORDER_2
        if out_deg <= 4:
            return self.OUTDEG_FILL_3_4, self.OUTDEG_BORDER_3_4
        return self.OUTDEG_FILL_5, self.OUTDEG_BORDER_5

    def render(self):
        self.screen.fill(self.BG)
        if not self.keys:
            t = self.font.render(
                f"No FSM data in {self.path}", True, self.TEXT,
            )
            self.screen.blit(t, (20, 20))
            pygame.display.flip()
            return
        key = self.keys[self.idx]
        fsm = self.fsms[key]
        # All display paths use the filtered view (stun stripped +
        # rare-edge pruned). Junctions are recomputed from filtered
        # out-degree so count and visible coloring agree; the saved
        # "junctions" field reflects the unfiltered graph and would
        # over-count by both the stun edges and the rare interrupts.
        transitions = filtered_transitions(
            fsm, edge_pct=self.edge_pct_threshold,
        )
        current = fsm.get("current_state")
        if current in STUN_STATES:
            current = None
        in_progress = [
            s for s in fsm.get("in_progress_sequence", [])
            if s not in STUN_STATES
        ]
        in_progress_set = set(in_progress)
        # In replay mode the displayed "current" is the state recorded at
        # the replay cursor — overrides the live current_state from the
        # save file so scrubbing through history lights up the right node.
        if self.replay_mode:
            replay_state = self.replay_state_at(self.replay_tick)
            if replay_state is not None:
                current = replay_state
            else:
                current = None
            in_progress_set = set()  # replay doesn't track in-progress chain
        g = self.graphs.get(key)
        if g is None:
            pygame.display.flip()
            return

        # Clip graph drawing to the left region so it doesn't paint over
        # the side panel.
        graph_rect = pygame.Rect(0, 0, self.graph_w, self.h)
        self.screen.set_clip(graph_rect)

        node_r = max(3, int(round(self.NODE_RADIUS * self.zoom)))
        head_size = max(4, int(round(10 * self.zoom)))
        edge_width = max(1, int(round(2 * self.zoom)))
        show_labels = self.zoom >= 0.55

        for s, dsts in transitions.items():
            if s not in g.positions:
                continue
            sx, sy = self.world_to_screen(*g.positions[s])
            for d in dsts:
                if d not in g.positions:
                    continue
                ex, ey = self.world_to_screen(*g.positions[d])
                if s == current:
                    color = self.EDGE_FROM_CURRENT
                elif d == current:
                    color = self.EDGE_TO_CURRENT
                else:
                    color = self.EDGE
                draw_arrow(
                    self.screen, color, (sx, sy), (ex, ey),
                    node_r, head=head_size, width=edge_width,
                )

        for n, (wx, wy) in g.positions.items():
            sx, sy = self.world_to_screen(wx, wy)
            if sx < -node_r * 4 or sx > self.graph_w + node_r * 4:
                continue
            if sy < -node_r * 4 or sy > self.h + node_r * 4:
                continue
            if n == current:
                fill, border = self.CURRENT_FILL, self.CURRENT_BORDER
            elif n in in_progress_set:
                fill, border = self.IN_PROGRESS_FILL, self.IN_PROGRESS_BORDER
            else:
                fill, border = self._outdeg_color(len(transitions.get(n, [])))
            pygame.draw.circle(self.screen, fill, (int(sx), int(sy)), node_r)
            pygame.draw.circle(self.screen, border, (int(sx), int(sy)), node_r, 2)
            if show_labels:
                label = self.label_font.render(n, True, self.TEXT)
                lx = int(sx) + node_r + 3
                ly = int(sy) - label.get_height() // 2
                bg = pygame.Surface(
                    (label.get_width() + 4, label.get_height()),
                    pygame.SRCALPHA,
                )
                bg.fill((255, 255, 255, 200))
                self.screen.blit(bg, (lx - 2, ly))
                self.screen.blit(label, (lx, ly))

        self.screen.set_clip(None)

        # Replay timeline strip (drawn before the header so the header
        # text overlays cleanly when the window is short).
        if self.replay_mode:
            self._draw_timeline(fsm)

        header = (
            f"epoch={self.epoch}    "
            f"FSM [{self.idx + 1}/{len(self.keys)}]: {key}"
        )
        if self.replay_mode:
            total = self.replay_total()
            play = "▶" if self.replay_playing else "∥"
            replay_state = current if current is not None else "(none)"
            header += (
                f"    REPLAY {play} tick={self.replay_tick}/{total}"
                f" speed={self.replay_speed:g}x state={replay_state}"
            )
        self.screen.blit(
            self.title_font.render(header, True, self.TEXT), (20, 10),
        )
        hint = (
            "LMB drag/pan  wheel zoom  ←/→ FSM  R reset  L reload  "
            "T replay  Space play  ,/. step  [/] " +
            ("speed" if self.replay_mode else "edge filter") +
            "  +/- attack filter  Q quit"
        )
        self.screen.blit(self.font.render(hint, True, self.DIM), (20, 30))
        # Recount junctions from filtered transitions so the stat
        # matches what's on screen (the saved "junctions" field counts
        # against the unfiltered graph and over-reports by the stun
        # edges we just hid).
        n_junctions = sum(1 for dsts in transitions.values() if len(dsts) >= 2)
        sub = (
            f"nodes={len(g.positions)}  "
            f"edges={sum(len(d) for d in transitions.values())}  "
            f"junctions={n_junctions}  "
            f"edge_filter=≥{self.edge_pct_threshold*100:.0f}%  "
            f"attacks={len(fsm.get('fingerprints', []))}  "
            f"zoom={self.zoom:.2f}"
        )
        self.screen.blit(self.font.render(sub, True, self.DIM), (20, 48))

        # Side panel.
        px = self.w - self.PANEL_W
        pygame.draw.line(
            self.screen, self.PANEL_RULE,
            (px - 6, 0), (px - 6, self.h), 1,
        )
        pygame.draw.rect(
            self.screen, self.PANEL_BG,
            pygame.Rect(px - 6, 0, self.PANEL_W, self.h),
        )
        y = 12
        self.screen.blit(
            self.title_font.render("LEGEND", True, self.TEXT), (px, y),
        )
        y += 24
        for fill, border, txt in [
            (self.CURRENT_FILL, self.CURRENT_BORDER, "current state"),
            (self.IN_PROGRESS_FILL, self.IN_PROGRESS_BORDER, "in-progress sequence"),
            (self.OUTDEG_FILL_1, self.OUTDEG_BORDER_1, "1 successor (linear)"),
            (self.OUTDEG_FILL_2, self.OUTDEG_BORDER_2, "2 successors"),
            (self.OUTDEG_FILL_3_4, self.OUTDEG_BORDER_3_4, "3-4 successors"),
            (self.OUTDEG_FILL_5, self.OUTDEG_BORDER_5, "5+ successors"),
        ]:
            pygame.draw.circle(self.screen, fill, (px + 10, y + 6), 8)
            pygame.draw.circle(self.screen, border, (px + 10, y + 6), 8, 2)
            self.screen.blit(self.font.render(txt, True, self.TEXT), (px + 28, y))
            y += 18
        y += 6
        thr = self.attack_pct_threshold
        all_fps = fsm.get("fingerprints", [])
        total_obs = sum(fp.get("count", 0) for fp in all_fps)
        # Scale-independent filter: an attack passes if its share of
        # total observations is at least `thr`. total_obs == 0 means
        # we're looking at a pre-count JSON; in that case the filter
        # is a no-op and everything renders.
        def passes(fp):
            if all(s in STUN_STATES for s in fp.get("sequence", [])):
                return False
            if total_obs <= 0:
                return True
            return fp.get("count", 0) / total_obs >= thr

        kept_n = sum(1 for fp in all_fps if passes(fp))
        title = f"ATTACKS  ≥{thr*100:.1f}%  ({kept_n}/{len(all_fps)})"
        self.screen.blit(
            self.title_font.render(title, True, self.TEXT), (px, y),
        )
        y += 22
        active_ids = set(fsm.get("active_ids", []))
        # Sort attacks by observation count desc so canonical attacks
        # surface to the top and the long tail of one-off interruption
        # variants sinks. Stable secondary sort on attack id keeps
        # equal-count entries in catalog order.
        ranked = sorted(
            (fp for fp in all_fps if passes(fp)),
            key=lambda fp: (-fp.get("count", 0),
                            int(fp.get("id", "a0")[1:])),
        )
        max_count = max((fp.get("count", 0) for fp in ranked), default=0)
        count_w = max(1, len(str(max_count)))
        for fp in ranked:
            atk_id = fp.get("id", "?")
            seq = [s for s in fp.get("sequence", []) if s not in STUN_STATES]
            count = fp.get("count", 0)
            pct = (count / total_obs * 100) if total_obs > 0 else 0.0
            active = atk_id in active_ids
            color = self.CURRENT_BORDER if active else self.TEXT
            marker = "▶" if active else " "
            text = f"{marker} {pct:4.1f}%  {atk_id}: {' → '.join(seq)}"
            max_w = self.PANEL_W - 10
            line_h = self.font.get_height() + 1
            line = text
            while line:
                surf = self.font.render(line, True, color)
                if surf.get_width() <= max_w:
                    self.screen.blit(surf, (px, y))
                    y += line_h
                    break
                lo, hi = 1, len(line)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self.font.size(line[:mid])[0] <= max_w:
                        lo = mid
                    else:
                        hi = mid - 1
                cut = lo
                surf = self.font.render(line[:cut], True, color)
                self.screen.blit(surf, (px, y))
                y += line_h
                line = "  " + line[cut:]
                if y > self.h - line_h:
                    break
            if y > self.h - line_h:
                break

        pygame.display.flip()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "state_graphs/latest.json"
    v = Viewer(path)
    v.run()


if __name__ == "__main__":
    main()
