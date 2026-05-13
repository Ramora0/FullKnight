import pygame

from env.observation import Observation, GS, CB, TR
from fsm_tracker import FsmTracker  # noqa: F401  (re-exported for back-compat)


class Visualizer:
    """Live pygame visualization of env 0's observation.

    Knight at origin, combat hitboxes color-coded by behavioral flags,
    terrain segments as line strokes with nearest-point dots.
    """

    # Window is split into the world canvas on the left and a fixed-width
    # FSM/info panel on the right. WORLD_W controls the canvas; PANEL_W is
    # added on top so the world layout is unaffected when the panel grows.
    WORLD_W = 1200
    PANEL_W = 460
    WIDTH = WORLD_W + PANEL_W
    HEIGHT = 720
    SCALE = 6.0  # pixels per world unit

    BG = (245, 245, 245)
    GRID = (225, 225, 225)
    AXIS = (180, 180, 220)
    KNIGHT_FILL = (130, 220, 230)
    KNIGHT_EDGE = (40, 70, 200)
    VEL = (40, 70, 200)
    TERRAIN = (20, 20, 20)
    TERRAIN_DIM = (210, 210, 210)  # filtered-out terrain (gate would drop)
    TRIGGER = (70, 130, 180)
    TRIGGER_DIM = (200, 215, 225)
    NEAREST = (0, 191, 255)
    VIEW_BOX = (90, 110, 200)
    TEXT = (35, 35, 35)

    # FSM panel palette. PANEL_BG is a faint gray so the panel reads as a
    # distinct region; PANEL_RULE separates sections. State text uses a
    # transition gradient: bright red on the frame a state changed, fading
    # to orange over the next few frames, then settling to plain text.
    PANEL_BG = (250, 250, 252)
    PANEL_RULE = (200, 200, 210)
    PANEL_HEADER = (30, 30, 80)
    PANEL_SECTION = (90, 50, 130)
    PANEL_DIM = (120, 120, 130)
    FSM_STATE = (35, 35, 35)
    FSM_CHANGED_HOT = (210, 30, 30)
    FSM_CHANGED_WARM = (220, 120, 30)
    FSM_DISPATCHER = (110, 110, 180)  # dim purple for "boss is choosing"
    FSM_ATTACK = (20, 100, 30)        # green for the current attack label
    # How many frames a state change stays visually highlighted. The
    # visualizer ticks at agent-step rate (~1/frames_per_wait of game time),
    # so 4 ticks ≈ a half-second pulse.
    FSM_HOT_FRAMES = 1
    FSM_WARM_FRAMES = 4

    # Combat colors mirror the matplotlib version:
    # red=target, orange=damageable enemy, magenta=hazard,
    # green=peaceful damageable, yellow=knight attack.
    COLOR_TARGET   = (220,  40,  40)
    COLOR_ENEMY    = (255, 140,  20)
    COLOR_HAZARD   = (220,  60, 200)
    COLOR_PEACEFUL = ( 60, 180,  60)
    COLOR_ATTACK   = (235, 215,  20)

    def __init__(self, vocab=None, terrain_max_dist=None, view_w=None, view_h=None, tracker=None):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("FullKnight Observation Viewer")
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.SysFont("consolas", 11)
        self.fsm_font = pygame.font.SysFont("consolas", 12)
        self.fsm_state_font = pygame.font.SysFont("consolas", 12, bold=True)
        self.title_font = pygame.font.SysFont("consolas", 14, bold=True)
        self.vocab = vocab
        # Optional terrain-gating preview: segments outside the gate are still
        # drawn but dimmed, so the user can eyeball how much is filtered before
        # baking the gate into the C# observer.
        self.terrain_max_dist = terrain_max_dist
        self.view_w = view_w
        self.view_h = view_h
        self._closed = False
        # World canvas is centered inside the LEFT WORLD_W pixels; the right
        # PANEL_W pixels are reserved for FSM text. Keeping cx tied to WORLD_W
        # rather than WIDTH means the panel doesn't shift the knight off-axis.
        self._cx = self.WORLD_W // 2
        self._cy = self.HEIGHT // 2
        # All FSM-tracking state lives on the tracker so the trainer can
        # record/save graph data without a live pygame visualizer.
        self.tracker = tracker if tracker is not None else FsmTracker()

    def _w2s(self, x, y):
        return self._cx + int(x * self.SCALE), self._cy - int(y * self.SCALE)

    def _world_rect(self, cx, cy, w, h):
        sx = self._cx + int((cx - w / 2) * self.SCALE)
        sy = self._cy - int((cy + h / 2) * self.SCALE)
        sw = max(1, int(w * self.SCALE))
        sh = max(1, int(h * self.SCALE))
        return pygame.Rect(sx, sy, sw, sh)

    def _combat_color(self, gives, takes, is_target):
        if is_target:
            return self.COLOR_TARGET
        if gives and takes:
            return self.COLOR_ENEMY
        if gives:
            return self.COLOR_HAZARD
        if takes:
            return self.COLOR_PEACEFUL
        return self.COLOR_ATTACK

    def update(self, obs: Observation, terrain_debug=None, fsm_snapshots=None):
        if self._closed:
            return
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._closed = True
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                self._closed = True
                return

        screen = self.screen
        screen.fill(self.BG)

        # Grid + origin axes
        for x in range(-100, 101, 10):
            sx, _ = self._w2s(x, 0)
            pygame.draw.line(screen, self.GRID, (sx, 0), (sx, self.HEIGHT))
        for y in range(-50, 51, 10):
            _, sy = self._w2s(0, y)
            pygame.draw.line(screen, self.GRID, (0, sy), (self.WIDTH, sy))
        pygame.draw.line(screen, self.AXIS, (0, self._cy), (self.WIDTH, self._cy))
        pygame.draw.line(screen, self.AXIS, (self._cx, 0), (self._cx, self.HEIGHT))

        gs = obs.global_state[0]
        vel_x = float(gs[GS.VEL_X]); vel_y = float(gs[GS.VEL_Y])
        hp = float(gs[GS.HP])
        knight_w = float(gs[GS.KNIGHT_W]); knight_h = float(gs[GS.KNIGHT_H])

        # Terrain segments. Two-pass: dimmed (gate-filtered) first so kept
        # segments paint on top.
        t_hb = obs.terrain_hb[0]
        t_mask = obs.terrain_mask[0]
        gate_active = (
            self.terrain_max_dist is not None
            or (self.view_w is not None and self.view_h is not None)
        )
        terrain_total = 0
        terrain_kept = 0
        kept_rows = []
        for i in range(len(t_mask)):
            if t_mask[i] < 0.5:
                continue
            row = t_hb[i]
            terrain_total += 1
            mx = float(row[TR.MX]); my = float(row[TR.MY])
            hdx = float(row[TR.HDX]); hdy = float(row[TR.HDY])
            npx = float(row[TR.NPX]); npy = float(row[TR.NPY])
            is_trigger = row[TR.IS_TRIGGER] > 0.5

            keep = True
            if self.terrain_max_dist is not None:
                if float(row[TR.DIST]) > self.terrain_max_dist:
                    keep = False
            if keep and self.view_w is not None and self.view_h is not None:
                if abs(npx) > self.view_w / 2 or abs(npy) > self.view_h / 2:
                    keep = False

            if keep:
                terrain_kept += 1
                kept_rows.append((mx, my, hdx, hdy, npx, npy, is_trigger))
            else:
                # Dim style for gate-dropped segments
                color = self.TRIGGER_DIM if is_trigger else self.TERRAIN_DIM
                pygame.draw.line(
                    screen, color,
                    self._w2s(mx - hdx, my - hdy),
                    self._w2s(mx + hdx, my + hdy),
                    1,
                )

        # Camera-box outline when a view-box gate is set
        if self.view_w is not None and self.view_h is not None:
            box_rect = self._world_rect(0, 0, self.view_w, self.view_h)
            pygame.draw.rect(screen, self.VIEW_BOX, box_rect, 1)

        # Kept terrain on top, in normal style
        for (mx, my, hdx, hdy, npx, npy, is_trigger) in kept_rows:
            color = self.TRIGGER if is_trigger else self.TERRAIN
            pygame.draw.line(
                screen, color,
                self._w2s(mx - hdx, my - hdy),
                self._w2s(mx + hdx, my + hdy),
                2,
            )
            pygame.draw.circle(screen, self.NEAREST, self._w2s(npx, npy), 2)

        # Combat hitboxes — collect labels and draw after so text sits on top.
        c_hb = obs.combat_hb[0]
        c_mask = obs.combat_mask[0]
        c_kid = obs.combat_kind_ids[0]
        c_pid = obs.combat_parent_ids[0]
        labels = []
        for i in range(len(c_mask)):
            if c_mask[i] < 0.5:
                continue
            row = c_hb[i]
            rx = float(row[CB.REL_X]); ry = float(row[CB.REL_Y])
            w = float(row[CB.W]); h = float(row[CB.H])
            color = self._combat_color(
                row[CB.GIVES_DAMAGE] > 0.5,
                row[CB.TAKES_DAMAGE] > 0.5,
                row[CB.IS_TARGET] > 0.5,
            )
            rect = self._world_rect(rx, ry, w, h)
            fill = pygame.Surface(rect.size, pygame.SRCALPHA)
            fill.fill((*color, 90))
            screen.blit(fill, rect.topleft)
            pygame.draw.rect(screen, color, rect, 2)

            kid = int(c_kid[i]); pid = int(c_pid[i])
            if self.vocab is not None:
                vlen = len(self.vocab)
                kname = self.vocab._i2s[kid] if 0 <= kid < vlen else str(kid)
                pname = self.vocab._i2s[pid] if 0 <= pid < vlen else str(pid)
                label = f"{kname}<{pname}>" if pid > 0 else kname
            else:
                label = f"{kid}<{pid}>" if pid > 0 else f"{kid}"
            if row[CB.TAKES_DAMAGE] > 0.5:
                label += f" hp={int(row[CB.HP_RAW])}/{int(row[CB.HP_MAX_RAW])}"
            labels.append((rect.topleft, label, color))

        # Knight at origin
        kr = self._world_rect(0, 0, knight_w, knight_h)
        kfill = pygame.Surface(kr.size, pygame.SRCALPHA)
        kfill.fill((*self.KNIGHT_FILL, 130))
        screen.blit(kfill, kr.topleft)
        pygame.draw.rect(screen, self.KNIGHT_EDGE, kr, 2)

        # Velocity arrow
        if abs(vel_x) > 0.01 or abs(vel_y) > 0.01:
            pygame.draw.line(
                screen, self.VEL,
                self._w2s(0, 0), self._w2s(vel_x, vel_y), 2,
            )

        # Combat labels on top
        for (lx, ly), text, color in labels:
            tsurf = self.font.render(text, True, self.TEXT)
            bg = pygame.Surface((tsurf.get_width() + 4, tsurf.get_height() + 2), pygame.SRCALPHA)
            bg.fill((*color, 200))
            screen.blit(bg, (lx, ly - tsurf.get_height() - 2))
            screen.blit(tsurf, (lx + 2, ly - tsurf.get_height() - 1))

        # Title bar
        terrain_str = (
            f"{terrain_kept}/{terrain_total}" if gate_active else f"{terrain_total}"
        )
        gate_bits = []
        if self.terrain_max_dist is not None:
            gate_bits.append(f"dist≤{self.terrain_max_dist:g}")
        if self.view_w is not None and self.view_h is not None:
            gate_bits.append(f"box {self.view_w:g}×{self.view_h:g}")
        gate_str = f"   gate: {', '.join(gate_bits)}" if gate_bits else ""
        title = (
            f"HP: {hp:.0f}   "
            f"Combat: {int(c_mask.sum())}   Terrain: {terrain_str}   "
            f"Vel: ({vel_x:.1f}, {vel_y:.1f}){gate_str}"
        )
        screen.blit(self.title_font.render(title, True, self.TEXT), (8, 6))

        # Feed the tracker if the caller handed us a snapshot — train.py
        # always feeds the tracker directly when vis is None, so we only
        # do this when vis IS the entry point (back-compat path).
        if fsm_snapshots is not None:
            self.tracker.update(fsm_snapshots)
        # FSM panel on the right. Drawn last so it sits above whatever combat
        # labels happen to cross into the panel region.
        self._render_fsm_panel()

        pygame.display.flip()

    def _render_wrapped_attack(self, x, y, head, seq, color, line_h, max_w):
        """Render `head + s1→s2→...→sn` wrapping at → boundaries.

        First line uses `head` as prefix; continuation lines use a hanging
        indent the width of `head` so the sequence aligns. Returns the y
        coordinate AFTER the rendered block (one line_h past the last line).
        """
        if not seq:
            self.screen.blit(self.fsm_font.render(head + "(empty)", True, color), (x, y))
            return y + line_h
        head_w = self.fsm_font.size(head)[0]
        indent = " " * max(0, int(head_w / self.fsm_font.size(" ")[0]))
        # Greedily pack parts into lines.
        parts = list(seq)
        lines = []
        current = [parts[0]]
        for p in parts[1:]:
            trial = current + [p]
            prefix = head if not lines else indent
            if self.fsm_font.size(prefix + "→".join(trial))[0] <= max_w:
                current = trial
            else:
                lines.append(current)
                current = [p]
        if current:
            lines.append(current)
        for i, lp in enumerate(lines):
            prefix = head if i == 0 else indent
            text = prefix + "→".join(lp) + ("→" if i < len(lines) - 1 else "")
            self.screen.blit(self.fsm_font.render(text, True, color), (x, y))
            y += line_h
            if y > self.HEIGHT - line_h:
                return y
        return y

    def _render_fsm_panel(self):
        """Render the right-hand FSM panel from the tracker's cached groups.

        Parsing happens in FsmTracker.update; we just read.
        """
        groups = self.tracker.last_groups
        empty_ticks = self.tracker.empty_ticks
        on_wire = self.tracker.last_on_wire

        # Panel background + separator rule.
        panel_x = self.WORLD_W
        pygame.draw.rect(
            self.screen, self.PANEL_BG,
            pygame.Rect(panel_x, 0, self.PANEL_W, self.HEIGHT),
        )
        pygame.draw.line(
            self.screen, self.PANEL_RULE,
            (panel_x, 0), (panel_x, self.HEIGHT), 1,
        )

        x_label = panel_x + 10
        y = 10
        total = sum(len(v) for v in groups.values())
        header = f"FSM SNAPSHOTS  ({total} parsed / {on_wire} on wire)"
        self.screen.blit(
            self.title_font.render(header, True, self.PANEL_HEADER),
            (x_label, y),
        )
        y += 22

        # Two failure modes that look identical in the per-section "(none)"
        # output:
        #   (a) wire delivered zero entries — likely an old mod DLL still
        #       loaded in HK (mod DLLs load once at HK startup; rebuild +
        #       restart HK to pick up the new code).
        #   (b) wire delivered entries but parse rejected them all (delimiter
        #       mismatch or unrecognized src tag).
        # The header above shows both numbers; the warning below names
        # whichever case we're in so the fix is obvious from the panel.
        if empty_ticks > 5:
            warn = "NO FSM DATA ON WIRE — rebuild mod + restart HK"
            self.screen.blit(
                self.fsm_state_font.render(warn, True, self.FSM_CHANGED_HOT),
                (x_label, y),
            )
            y += 16
        elif on_wire > 0 and total == 0:
            warn = f"PARSE FAIL — {on_wire} entries arrived but none matched"
            self.screen.blit(
                self.fsm_state_font.render(warn, True, self.FSM_CHANGED_HOT),
                (x_label, y),
            )
            y += 16

        legend = "B=boss subtree   E=enemy hitbox   A=knight hitbox"
        self.screen.blit(
            self.font.render(legend, True, self.PANEL_DIM),
            (x_label, y),
        )
        y += 16
        pygame.draw.line(
            self.screen, self.PANEL_RULE,
            (panel_x + 6, y), (panel_x + self.PANEL_W - 6, y), 1,
        )
        y += 6

        section_titles = {
            "B": "BOSS SUBTREE FSMs  (src=B)",
            "E": "ENEMY HITBOX FSMs  (src=E)",
            "A": "KNIGHT HITBOX FSMs (src=A)",
        }
        line_h = self.fsm_font.get_height() + 1
        for src in ("B", "E", "A"):
            rows = groups[src]
            title_text = f"{section_titles[src]}   [{len(rows)}]"
            self.screen.blit(
                self.fsm_state_font.render(title_text, True, self.PANEL_SECTION),
                (x_label, y),
            )
            y += line_h + 1
            if not rows:
                self.screen.blit(
                    self.fsm_font.render("  (none)", True, self.PANEL_DIM),
                    (x_label, y),
                )
                y += line_h + 4
                continue
            # Sort by owner then fsm so the panel reads stable frame-to-frame
            # even if HK rebuilt the FSM list in a different order. Ties at
            # the source level keep their group; we just sort within group.
            rows.sort(key=lambda r: (r[0], r[1]))
            for owner, fsm_name, state, key, state_is_junction in rows:
                age = self.tracker._changed_age.get(key, self.tracker.FSM_WARM_FRAMES)
                if age <= self.FSM_HOT_FRAMES:
                    state_color = self.FSM_CHANGED_HOT
                    marker = "▶"
                elif age <= self.FSM_WARM_FRAMES:
                    state_color = self.FSM_CHANGED_WARM
                    marker = "·"
                else:
                    state_color = self.FSM_STATE
                    marker = " "
                if state_is_junction:
                    # Junction states (observed out-deg ≥ 2) get a distinct
                    # visual — they're not attacks, they're the boss picking
                    # the next one. Overrides the transition pulse since the
                    # pulse meaning ("attack started") doesn't apply at hubs.
                    state_color = self.FSM_DISPATCHER
                    marker = "◇"
                prefix = f"  {marker} {owner} | {fsm_name} → "
                state_text = state if state else "(none)"
                prefix_surf = self.fsm_font.render(prefix, True, self.FSM_STATE)
                state_surf = self.fsm_state_font.render(state_text, True, state_color)
                # Truncate if the combined line is wider than the panel.
                max_w = self.PANEL_W - 16
                if prefix_surf.get_width() + state_surf.get_width() > max_w:
                    # Drop the owner column first when space-constrained.
                    short_prefix = f"  {marker} {fsm_name} → "
                    prefix_surf = self.fsm_font.render(short_prefix, True, self.FSM_STATE)
                self.screen.blit(prefix_surf, (x_label, y))
                self.screen.blit(
                    state_surf,
                    (x_label + prefix_surf.get_width(), y),
                )
                y += line_h
                # Attack inventory readout — every fingerprint discovered for
                # this FSM gets its own row beneath the current-state line.
                # Long sequences wrap across multiple display lines with a
                # hanging indent. The currently-active ID (most recently
                # finalized segment) is highlighted in FSM_ATTACK; others
                # render in dim text. Iterates in mint order (a0, a1, ...).
                fp_map = self.tracker._fingerprint_to_id.get(key, {})
                active_set = self.tracker._active_ids.get(key, set())
                if fp_map:
                    items = sorted(fp_map.items(), key=lambda kv: int(kv[1][1:]))
                    max_w = self.PANEL_W - 16
                    for seq, atk_id in items:
                        is_active = atk_id in active_set
                        marker_a = "▶" if is_active else " "
                        color = self.FSM_ATTACK if is_active else self.PANEL_DIM
                        head = f"    {marker_a} {atk_id}: "
                        y = self._render_wrapped_attack(
                            x_label, y, head, seq, color, line_h, max_w,
                        )
                        if y > self.HEIGHT - line_h:
                            return
                if y > self.HEIGHT - line_h:
                    # Out of vertical room — drop remaining lines silently.
                    return
            y += 4

    def save_graph_state(self, filepath, epoch=None):
        """Back-compat delegate — actual save lives on FsmTracker."""
        self.tracker.save_graph_state(filepath, epoch=epoch)

    def close(self):
        if self._closed:
            return
        self._closed = True
        pygame.display.quit()
        pygame.font.quit()
