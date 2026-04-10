import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from observation import Observation, GS, CB


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

    def update(self, obs: Observation):
        """Redraw with the current Observation (env 0 only)."""
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
            rect = patches.Rectangle(
                (rx - w / 2, ry - h / 2), w, h,
                linewidth=1, edgecolor="gray", facecolor="lightgray", alpha=0.5,
            )
            ax.add_patch(rect)

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
                label += f" hp={int(hp_raw)}"
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
