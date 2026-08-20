# visualize.py
# Handles all 2D animation and plotting — no physics here, just drawing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from constants import R_EARTH
from physics import rk4_step, orbital_energy, orbital_period

def compute_orbit_points(position, velocity, dt=10, max_time=None, max_points=10000):
    """
    Run the RK4 integrator and collect position points for the full orbit.
    Returns arrays of x, y positions and the corresponding velocities.
    """
    if max_time is None:
        # Default: simulate for one full orbital period
        r = np.linalg.norm(position)
        max_time = orbital_period(r - R_EARTH)

    positions  = [position.copy()]
    velocities = [velocity.copy()]

    pos = position.copy()
    vel = velocity.copy()
    t   = 0

    while t < max_time and len(positions) < max_points:
        pos, vel = rk4_step(pos, vel, dt)
        positions.append(pos.copy())
        velocities.append(vel.copy())
        t += dt

        # Stop if satellite hits Earth
        if np.linalg.norm(pos) < R_EARTH:
            break

    return np.array(positions), np.array(velocities)


def draw_earth(ax):
    """Draw Earth as a filled circle to scale."""
    earth = Circle((0, 0), R_EARTH, color='deepskyblue', zorder=2, label='Earth')
    ax.add_patch(earth)

    # Add a simple atmosphere glow
    atmosphere = Circle((0, 0), R_EARTH + 100000,
                        color='deepskyblue', alpha=0.1, zorder=1)
    ax.add_patch(atmosphere)


def animate_orbit(position, velocity, orbit_type, dt=10):
    """
    Main animation function.
    Shows the satellite moving along its orbit in real time.
    """
    print("Computing orbit trajectory...")
    positions, velocities = compute_orbit_points(position, velocity, dt=dt)

    # --- Figure setup ---
    fig, (ax_orbit, ax_stats) = plt.subplots(
        1, 2,
        figsize=(14, 7),
        gridspec_kw={'width_ratios': [2, 1]}
    )
    fig.patch.set_facecolor('#0a0a1a')

    # --- Orbit panel ---
    ax_orbit.set_facecolor('#0a0a1a')
    ax_orbit.set_aspect('equal')
    ax_orbit.tick_params(colors='white')
    ax_orbit.xaxis.label.set_color('white')
    ax_orbit.yaxis.label.set_color('white')
    for spine in ax_orbit.spines.values():
        spine.set_edgecolor('#333355')

    # Draw Earth
    draw_earth(ax_orbit)

    # Draw the full orbit path faintly
    ax_orbit.plot(positions[:, 0], positions[:, 1],
                  color='#334466', linewidth=1, zorder=3)

    # Set axis limits with padding
    max_range = np.max(np.abs(positions[:, :2])) * 1.2
    ax_orbit.set_xlim(-max_range, max_range)
    ax_orbit.set_ylim(-max_range, max_range)
    ax_orbit.set_xlabel('x (m)', color='white')
    ax_orbit.set_ylabel('y (m)', color='white')
    ax_orbit.set_title(f'Orbit Simulation — {orbit_type.capitalize()}',
                       color='white', fontsize=13)

    # Satellite dot and trail
    satellite_dot,  = ax_orbit.plot([], [], 'o',
                                    color='yellow', markersize=6, zorder=5)
    trail_line,     = ax_orbit.plot([], [], '-',
                                    color='orange', linewidth=1.2,
                                    alpha=0.6, zorder=4)

    # Apoapsis and periapsis markers
    distances = np.linalg.norm(positions, axis=1)
    apo_idx  = np.argmax(distances)
    peri_idx = np.argmin(distances)

    ax_orbit.plot(*positions[apo_idx, :2],  'v',
                  color='lime',   markersize=8, label='Apoapsis',  zorder=6)
    ax_orbit.plot(*positions[peri_idx, :2], '^',
                  color='red',    markersize=8, label='Periapsis', zorder=6)
    ax_orbit.legend(facecolor='#1a1a2e', labelcolor='white',
                    loc='upper right', fontsize=8)

    # --- Stats panel ---
    ax_stats.set_facecolor('#0a0a1a')
    ax_stats.axis('off')
    ax_stats.set_title('Mission Stats', color='white', fontsize=12, pad=10)

    initial_energy = orbital_energy(positions[0], velocities[0])

    stat_text = ax_stats.text(
        0.05, 0.95, '', transform=ax_stats.transAxes,
        color='white', fontsize=10, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8)
    )

    # --- Animation update function ---
    trail_length = 60  # number of past frames to show as trail

    def update(frame):
        idx = frame % len(positions)

        # Move satellite dot
        x, y = positions[idx, 0], positions[idx, 1]
        satellite_dot.set_data([x], [y])

        # Draw trail
        start = max(0, idx - trail_length)
        trail_line.set_data(positions[start:idx, 0],
                            positions[start:idx, 1])

        # Current stats
        pos = positions[idx]
        vel = velocities[idx]
        r   = np.linalg.norm(pos)
        v   = np.linalg.norm(vel)
        alt = (r - R_EARTH) / 1000
        t   = idx * dt / 60

        energy      = orbital_energy(pos, vel)
        energy_drift = abs(energy - initial_energy) / abs(initial_energy) * 100 # howmuch energy has been lost, should be very tiny percent

        apo_alt  = (distances[apo_idx]  - R_EARTH) / 1000
        peri_alt = (distances[peri_idx] - R_EARTH) / 1000

        stats = (
            f"  Time          : {t:.1f} min\n\n"
            f"  Altitude      : {alt:.1f} km\n"
            f"  Velocity      : {v/1000:.3f} km/s\n\n"
            f"  Apoapsis      : {apo_alt:.1f} km\n"
            f"  Periapsis     : {peri_alt:.1f} km\n\n"
            f"  Orbit type    : {orbit_type}\n\n"
            f"  Energy drift  : {energy_drift:.6f}%\n"
        )
        stat_text.set_text(stats)

        return satellite_dot, trail_line, stat_text

    # --- Run animation ---
    interval_ms = 20  # milliseconds between frames
    ani = animation.FuncAnimation(
        fig, update,
        frames=len(positions),
        interval=interval_ms,
        blit=True
    )

    plt.tight_layout()
    plt.show()
    return ani