import numpy as np
from constants import R_EARTH
from physics import orbital_velocity, orbital_period, escape_velocity, orbital_energy

#This will get parameters from user
def get_user_input(): 
    print("\n=== Orbital Mechanics Simulator ===\n")

    while True:
        try:
            altitude_km = float(input("Enter orbital altitude in km: "))
            if altitude_km < 80:
                print("Too low, below reentry threshold. Try above 80 km.")
            elif altitude_km > 100000:
                print("Too high, try below 100,000 km.")
            else:
                break
        except ValueError:
            print("Please enter a valid number.")
    
    altitude_m = altitude_km * 1000

    circ_v = orbital_velocity(altitude_m)
    esc_v = escape_velocity(altitude_m)

    print(f"\nAt {altitude_km} km altitude:")
    print(f"Circular Orbit Velocity: {circ_v/1000:.2f} km/s")
    print(f"Escape Velocity: {esc_v/1000:.2f} km/s")
    print(f"\nHelpful Notes for inputting velocity:\nbelow circular = crash\nbetween = elliptical orbit\nabove escape = leave Earth")

    while True:
        try:
            velocity_kms = float(input("\nEnter initial velocity (km/s): "))
            if velocity_kms <= 0:
                print("  Velocity must be positive.")
            else:
                break
        except ValueError:
            print("  Please enter a valid number.")

    velocity_ms = velocity_kms * 1000
    return altitude_m, velocity_ms


def classify_orbit(velocity_ms, altitude_m):
    """Thiss will tell the user what kind of trajectory they have entered."""
    circ_v = orbital_velocity(altitude_m)
    esc_v  = escape_velocity(altitude_m)
    tolerance = 1e-3
    relative_error = abs(velocity_ms - circ_v) / circ_v

    if velocity_ms < 0.95 * circ_v:
        return "suborbital"         # will crash
    elif relative_error < tolerance:
        return "circular"           # roughly circular
    elif velocity_ms < esc_v:
        return "elliptical"         # ellipcatial orbit
    else:
        return "escape"             # leaves Earth

def setup_initial_conditions(altitude_m, velocity_ms):
    """
    Place satellite at (r, 0, 0) and it will move in the +y direction.
    So this is the standard starting position, due right of Earth,
    moving upward. All orbits will start here regardless of shape.
    """
    r = R_EARTH + altitude_m
    position = np.array([r, 0.0, 0.0])
    velocity = np.array([0.0, velocity_ms, 0.0])
    return position, velocity

def print_mission_summary(altitude_m, velocity_ms):
    """Printing a summary of the mission parameters."""
    orbit_type = classify_orbit(velocity_ms, altitude_m)
    period = orbital_period(altitude_m)
    circ_v = orbital_velocity(altitude_m)

    r = R_EARTH + altitude_m
    pos, vel = setup_initial_conditions(altitude_m, velocity_ms)
    energy = orbital_energy(pos, vel)

    print("\n--- Mission Summary ---")
    print(f"  Altitude         : {altitude_m/1000:.1f} km")
    print(f"  Initial velocity : {velocity_ms/1000:.2f} km/s")
    print(f"  Circular v       : {circ_v/1000:.2f} km/s")
    print(f"  Orbit type       : {orbit_type}")
    print(f"  Orbital period   : {period/60:.1f} minutes")
    print(f"  Orbital energy   : {energy:.2f} J/kg")
    print("-----------------------\n")

    return orbit_type


if __name__ == "__main__":
    altitude_m, velocity_ms  = get_user_input()
    orbit_type               = print_mission_summary(altitude_m, velocity_ms)
    position, velocity       = setup_initial_conditions(altitude_m, velocity_ms)

    print(f"  Launching simulation now...\n")

    from visualize import animate_orbit
    ani = animate_orbit(position, velocity, orbit_type)