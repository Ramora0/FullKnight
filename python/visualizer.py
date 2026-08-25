import pygame

from observation import Observation, GS, CB, TR


class Visualizer:
    """Live pygame visualization of env 0's observation.

    Knight at origin, combat hitboxes color-coded by behavioral flags,
    terrain segments as line strokes with nearest-point dots.
    """

    WIDTH = 1200
    HEIGHT = 600
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

    # Combat colors mirror the matplotlib version:
    # red=target, orange=damageable enemy, magenta=hazard,
    # green=peaceful damageable, yellow=knight attack.
    COLOR_TARGET   = (220,  40,  40)
    COLOR_ENEMY    = (255, 140,  20)
    COLOR_HAZARD   = (220,  60, 200)
    COLOR_PEACEFUL = ( 60, 180,  60)
    COLOR_ATTACK   = (235, 215,  20)

    def __init__(self, vocab=None, terrain_max_dist=None, view_w=None,
                 view_h=None, anim_vocab=None):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("FullKnight Observation Viewer")
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.SysFont("consolas", 11)
        self.title_font = pygame.font.SysFont("consolas", 14, bold=True)
        self.vocab = vocab
        self.anim_vocab = anim_vocab
        # Optional terrain-gating preview: segments outside the gate are still
        # drawn but dimmed, so the user can eyeball how much is filtered before
        # baking the gate into the C# observer.
        self.terrain_max_dist = terrain_max_dist
        self.view_w = view_w
        self.view_h = view_h
        self._closed = False
        self._cx = self.WIDTH // 2
        self._cy = self.HEIGHT // 2

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

    def update(self, obs: Observation, terrain_debug=None):
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
        c_aid = obs.combat_anim_ids[0]
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
            if row[CB.IS_INVINCIBLE] > 0.5:
                label += " INV"
            # Animation clip + phase — the telegraph channel. Shown as
            # "clip:NN%" so a windup is visible ticking up before its attack
            # collider ever appears.
            aid = int(c_aid[i])
            if aid > 0:
                if self.anim_vocab is not None and 0 <= aid < len(self.anim_vocab):
                    aname = self.anim_vocab._i2s[aid]
                else:
                    aname = str(aid)
                label += f" {aname}:{row[CB.ANIM_PROGRESS] * 100:.0f}%"
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

        pygame.display.flip()

    def close(self):
        if self._closed:
            return
        self._closed = True
        pygame.display.quit()
        pygame.font.quit()
