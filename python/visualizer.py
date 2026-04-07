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

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("FullKnight Observation Viewer")

    def update(self, combat_hb, combat_mask, terrain_hb, terrain_mask, global_state):
        """Redraw with current observations (all batched numpy arrays, shows index 0)."""
        ax = self.ax
        ax.clear()

        gs = global_state[0]
        vel_x, vel_y = gs[0], gs[1]
        hp = gs[2]
        boss_hp = gs[5]
        knight_w = gs[6]
        knight_h = gs[7]

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

        # Combat hitboxes (red = non-trigger, orange = trigger)
        c_hb = combat_hb[0]
        c_mask = combat_mask[0]
        for i in range(len(c_mask)):
            if c_mask[i] < 0.5:
                continue
            rx, ry, w, h, is_trig = c_hb[i]
            color = "orange" if is_trig > 0.5 else "red"
            rect = patches.Rectangle(
                (rx - w / 2, ry - h / 2), w, h,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
            )
            ax.add_patch(rect)

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
        ax.autoscale()
        ax.margins(0.1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="blue", linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color="blue", linewidth=0.5, alpha=0.3)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)
