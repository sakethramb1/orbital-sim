import numpy as np
from constants import GM, R_EARTH

# altitude_m is altitiude in meters

def orbital_velocity(altitude_m): # v = sqrt(GM/R)
    r = R_EARTH + altitude_m
    return np.sqrt(GM/r)

def orbital_period(altitude_m):
    r = R_EARTH + altitude_m
    return 2 * np.pi * np.sqrt(r**3/GM)

def escape_velocity(altitude_m):
    r = R_EARTH + altitude_m
    return np.sqrt(2*GM / r)

def orbital_energy(position, velocity, mass=1.0):
    """
    Total specific orbital energy (kinetic + potential).
    Should stay constant throughout simulation if physics is correct.
    """
    r = np.linalg.norm(position) # just sqrt of x^2 + y^2 + z^2
    v = np.linalg.norm(velocity) # same
    kinetic = 0.5 * v**2
    potential = -GM / r
    return kinetic + potential

def acceleration(position):
    """
    Gravitational acceleration vector at a given position.
    a = GM/r² directed toward Earth's center
    """
    r = np.linalg.norm(position)
    return -GM * position / r**3

def rk4_step(position, velocity, dt):
    """
    Single RK4 integration step.
    Takes current position and velocity, returns new position and velocity
    after time dt (in seconds).
    """
    # K1 — slope at current point
    k1_v = acceleration(position) #basically v', what are the derivatives right now?
    k1_r = velocity #basically r'

    # K2 — slope at midpoint using k1
    k2_v = acceleration(position + 0.5 * dt * k1_r)
    k2_r = velocity + 0.5 * dt * k1_v

    # K3 — slope at midpoint using k2
    k3_v = acceleration(position + 0.5 * dt * k2_r)
    k3_r = velocity + 0.5 * dt * k2_v

    # K4 — slope at end point using k3
    k4_v = acceleration(position + dt * k3_r)
    k4_r = velocity + dt * k3_v

    # Weighted average of all four slopes
    new_position = position + (dt / 6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    new_velocity = velocity + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return new_position, new_velocity