import json
import os
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from observation import Observation, GS, CB


_TERRAIN_DEBUG_FIELDS = [
    "name", "path", "col_type", "layer",
    "enabled", "active", "trigger", "used_by_composite",
    "world_bounds",
]


def parse_terrain_debug(s: str) -> dict:
    """Split a pipe-delimited terrain_debug string into a dict.

    Layout:
      - First N fields are positional (see _TERRAIN_DEBUG_FIELDS).
      - Any remaining field of the form "k=v" is stored as out[k]=v.
      - The special "segments" key is expanded into a list of (x1,y1,x2,y2)
        tuples, the rest stay as strings (caller can cast if needed)."""
    if not s:
        return {}
    parts = s.split("|")
    positional = [p for p in parts if "=" not in p]
    kv = [p for p in parts if "=" in p]
    out: dict = {}
    for i, key in enumerate(_TERRAIN_DEBUG_FIELDS):
        if i < len(positional):
            out[key] = positional[i]
    for p in kv:
        k, _, v = p.partition("=")
        if k == "segments":
            segs = []
            for tri in v.split(";"):
                if not tri:
                    continue
                xs = tri.split(",")
                if len(xs) != 4:
                    continue
                try:
                    segs.append((float(xs[0]), float(xs[1]), float(xs[2]), float(xs[3])))
                except ValueError:
                    pass
            out["segments"] = segs
        else:
            out[k] = v
    return out


class Visualizer:
    """Live visualization of the exact observations the model receives.

    Shows env 0 only. Knight at origin, combat hitboxes in red/orange,
    terrain in gray, with global state info in the title.
    """

    def __init__(self, vocab=None):
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("FullKnight Observation Viewer")
        self.vocab = vocab
        # Latest inputs — kept so the 's' key handler can dump them.
        self._last_obs = None
        self._last_terrain_debug = []
        self._snapshot_dir = "debug_snapshots"
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def update(self, obs: Observation, terrain_debug=None):
        """Redraw with the current Observation (env 0 only)."""
        self._last_obs = obs
        self._last_terrain_debug = list(terrain_debug or [])
        ax = self.ax
        ax.clear()

        gs = obs.global_state[0]
        vel_x, vel_y = gs[GS.VEL_X], gs[GS.VEL_Y]
        hp = gs[GS.HP]
        knight_w = gs[GS.KNIGHT_W]
        knight_h = gs[GS.KNIGHT_H]

        # Terrain hitboxes (gray)
        t_hb = obs.terrain_hb[0]
        t_mask = obs.terrain_mask[0]
        for i in range(len(t_mask)):
            if t_mask[i] < 0.5:
                continue
            rx, ry, w, h, _ = t_hb[i]
            dbg = parse_terrain_debug(
                self._last_terrain_debug[i]
                if i < len(self._last_terrain_debug) else ""
            )
            # "Ghost" = physics engine won't let the knight collide with this.
            # Ghost boxes are tinted magenta so they jump out against the normal
            # gray terrain. Any one of these checks is enough to flag it.
            is_ghost = (
                dbg.get("layer_ignore") == "1"
                or dbg.get("pair_ignore") == "1"
                or dbg.get("used_by_composite") == "1"
                or dbg.get("trigger") == "1"
                or dbg.get("enabled") == "0"
                or dbg.get("active") == "0"
                or (dbg.get("rb") == "1" and dbg.get("rb_sim") == "0")
            )
            edge_col = "magenta" if is_ghost else "gray"
            face_col = "mistyrose" if is_ghost else "lightgray"
            rect = patches.Rectangle(
                (rx - w / 2, ry - h / 2), w, h,
                linewidth=1, edgecolor=edge_col, facecolor=face_col, alpha=0.3,
            )
            ax.add_patch(rect)
            # Overlay the true collision polyline on top of the AABB (solid
            # dark line) so the gap between the gray box the agent "sees"
            # and what it would actually collide with is visually obvious.
            segs = dbg.get("segments") or []
            for (x1, y1, x2, y2) in segs:
                ax.plot(
                    [x1, x2], [y1, y2],
                    color="black", linewidth=1.5, alpha=0.9, solid_capstyle="round",
                )
            label = dbg.get("name", "")
            flags = []
            if dbg.get("enabled") == "0": flags.append("dis")
            if dbg.get("active") == "0": flags.append("inact")
            if dbg.get("used_by_composite") == "1": flags.append("composite")
            if dbg.get("trigger") == "1": flags.append("trig")
            if dbg.get("layer_ignore") == "1": flags.append("layer!")
            if dbg.get("pair_ignore") == "1": flags.append("pair!")
            if dbg.get("rb") == "1" and dbg.get("rb_sim") == "0":
                flags.append("!sim")
            # ray_self=0 means something else sits between knight and this box.
            # Stamp the blocker name so the cause is visible inline.
            if dbg.get("ray_self") == "0":
                flags.append(f"blk:{dbg.get('ray_first','?')}")
            if dbg.get("touching") == "1": flags.append("touch")
            if segs:
                flags.append(f"{len(segs)}seg")
            if flags:
                label = f"{label} [{','.join(flags)}]"
            if label:
                ax.text(
                    rx - w / 2, ry - h / 2, label,
                    fontsize=6, color="black",
                    bbox=dict(facecolor="lightgray", alpha=0.6, edgecolor="none", pad=1),
                    verticalalignment="top", horizontalalignment="left",
                )

        # Combat hitboxes — colors encode the three behavioral flags:
        #   red    = boss target (gives + takes + is_target)
        #   orange = damageable enemy that's not the goal (gives + takes, no target)
        #   magenta = pure projectile / hazard (gives, no takes)
        #   green  = peaceful target (takes / target, no gives) — chests, exits, future
        #   yellow = knight's own attack (no gives, no takes)
        c_hb = obs.combat_hb[0]
        c_mask = obs.combat_mask[0]
        c_kid = obs.combat_kind_ids[0]
        c_pid = obs.combat_parent_ids[0]
        for i in range(len(c_mask)):
            if c_mask[i] < 0.5:
                continue
            row = c_hb[i]
            rx, ry, w, h = row[CB.REL_X], row[CB.REL_Y], row[CB.W], row[CB.H]
            gives = row[CB.GIVES_DAMAGE]
            takes = row[CB.TAKES_DAMAGE]
            is_target = row[CB.IS_TARGET]
            hp_raw = row[CB.HP_RAW]
            hp_max_raw = row[CB.HP_MAX_RAW]
            if is_target > 0.5:
                color = "red"
            elif gives > 0.5 and takes > 0.5:
                color = "orange"
            elif gives > 0.5:
                color = "magenta"
            elif takes > 0.5:
                color = "green"
            else:
                color = "yellow"
            rect = patches.Rectangle(
                (rx - w / 2, ry - h / 2), w, h,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
            )
            ax.add_patch(rect)

            # Kind+parent id label + raw HP if damageable, anchored to top-left of the box.
            kid = int(c_kid[i])
            pid = int(c_pid[i])
            if self.vocab is not None:
                kname = self.vocab._i2s[kid] if 0 <= kid < len(self.vocab) else str(kid)
                pname = self.vocab._i2s[pid] if 0 <= pid < len(self.vocab) else str(pid)
                label = f"{kname}<{pname}>" if pid > 0 else kname
            else:
                label = f"{kid}<{pid}>" if pid > 0 else f"{kid}"
            if takes > 0.5:
                label += f" hp={int(hp_raw)}/{int(hp_max_raw)}"
            ax.text(
                rx - w / 2, ry + h / 2, label,
                fontsize=7, color="black",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
                verticalalignment="bottom", horizontalalignment="left",
            )

        # Knight at origin
        knight_rect = patches.Rectangle(
            (-knight_w / 2, -knight_h / 2), knight_w, knight_h,
            linewidth=2, edgecolor="blue", facecolor="cyan", alpha=0.5,
        )
        ax.add_patch(knight_rect)

        # Velocity arrow
        if abs(vel_x) > 0.01 or abs(vel_y) > 0.01:
            ax.arrow(0, 0, vel_x, vel_y, head_width=0.15, head_length=0.1,
                     fc="blue", ec="blue", alpha=0.6)

        ax.set_title(
            f"HP: {hp:.0f}  "
            f"Combat: {int(c_mask.sum())}  Terrain: {int(t_mask.sum())}  "
            f"Vel: ({vel_x:.1f}, {vel_y:.1f})"
        )
        ax.set_aspect("equal")
        ax.set_xlim(-100, 100)
        ax.set_ylim(-50, 50)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="blue", linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color="blue", linewidth=0.5, alpha=0.3)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)

    def _on_key(self, event):
        if event.key == "s":
            path = self.save_snapshot()
            if path:
                print(f"[viewer] snapshot saved: {path}")

    def save_snapshot(self, path=None):
        """Dump the last observation + terrain debug to a JSON file.
        Returns the written path or None if there's nothing to save."""
        if self._last_obs is None:
            return None
        os.makedirs(self._snapshot_dir, exist_ok=True)
        if path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self._snapshot_dir, f"snapshot_{ts}.json")

        obs = self._last_obs
        gs = obs.global_state[0]
        payload: dict = {
            "global_state": {
                "vel_x": float(gs[GS.VEL_X]),
                "vel_y": float(gs[GS.VEL_Y]),
                "hp": float(gs[GS.HP]),
                "soul": float(gs[GS.SOUL]),
                "knight_w": float(gs[GS.KNIGHT_W]),
                "knight_h": float(gs[GS.KNIGHT_H]),
                "raw": [float(x) for x in gs],
            },
            "combat": [],
            "terrain": [],
        }

        c_hb = obs.combat_hb[0]
        c_mask = obs.combat_mask[0]
        c_kid = obs.combat_kind_ids[0]
        c_pid = obs.combat_parent_ids[0]
        for i in range(len(c_mask)):
            if c_mask[i] < 0.5:
                continue
            row = c_hb[i]
            kid = int(c_kid[i])
            pid = int(c_pid[i])
            kname = self.vocab._i2s[kid] if (self.vocab is not None and 0 <= kid < len(self.vocab._i2s)) else str(kid)
            pname = self.vocab._i2s[pid] if (self.vocab is not None and 0 <= pid < len(self.vocab._i2s)) else str(pid)
            payload["combat"].append({
                "kind": kname,
                "parent": pname,
                "rel_x": float(row[CB.REL_X]),
                "rel_y": float(row[CB.REL_Y]),
                "w": float(row[CB.W]),
                "h": float(row[CB.H]),
                "is_trigger": bool(row[CB.IS_TRIGGER] > 0.5),
                "gives_damage": bool(row[CB.GIVES_DAMAGE] > 0.5),
                "takes_damage": bool(row[CB.TAKES_DAMAGE] > 0.5),
                "is_target": bool(row[CB.IS_TARGET] > 0.5),
                "hp_raw": float(row[CB.HP_RAW]),
            })

        t_hb = obs.terrain_hb[0]
        t_mask = obs.terrain_mask[0]
        for i in range(len(t_mask)):
            if t_mask[i] < 0.5:
                continue
            rx, ry, w, h, trig = t_hb[i]
            dbg = parse_terrain_debug(
                self._last_terrain_debug[i]
                if i < len(self._last_terrain_debug) else ""
            )
            payload["terrain"].append({
                "rel_x": float(rx),
                "rel_y": float(ry),
                "w": float(w),
                "h": float(h),
                "is_trigger_obs": bool(trig > 0.5),
                "debug": dbg,
            })

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path
