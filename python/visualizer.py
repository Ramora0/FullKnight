import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


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

    def update(self, combat_hb, combat_mask, combat_kind_ids, terrain_hb, terrain_mask, global_state):
        """Redraw with current observations (all batched numpy arrays, shows index 0)."""
        ax = self.ax
        ax.clear()

        gs = global_state[0]
        vel_x, vel_y = gs[0], gs[1]
        hp = gs[2]
        boss_hp = gs[4]
        knight_w = gs[5]
        knight_h = gs[6]

        # Terrain hitboxes (gray)
        t_hb = terrain_hb[0]
        t_mask = terrain_mask[0]
        for i in range(len(t_mask)):
            if t_mask[i] < 0.5:
                continue
            rx, ry, w, h, _ = t_hb[i]
            rect = patches.Rectangle(
                (rx - w / 2, ry - h / 2), w, h,
                linewidth=1, edgecolor="gray", facecolor="lightgray", alpha=0.5,
            )
            ax.add_patch(rect)

        # Combat hitboxes: green = target (boss), red = hurts knight, yellow = knight's attack
        c_hb = combat_hb[0]
        c_mask = combat_mask[0]
        c_kid = combat_kind_ids[0]
        for i in range(len(c_mask)):
            if c_mask[i] < 0.5:
                continue
            rx, ry, w, h, is_trig, hurts_knight, is_target = c_hb[i]
            if is_target > 0.5:
                color = "red" if hurts_knight > 0.5 else "green"
            else:
                color = "yellow"
            rect = patches.Rectangle(
                (rx - w / 2, ry - h / 2), w, h,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
            )
            ax.add_patch(rect)

            # Kind id label, anchored to top-left of the box
            kid = int(c_kid[i])
            if self.vocab is not None and 0 <= kid < len(self.vocab):
                label = f"{kid}:{self.vocab._i2s[kid]}"
            else:
                label = f"{kid}"
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
            f"HP: {hp:.0f}  Boss HP: {boss_hp:.1f}  "
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
