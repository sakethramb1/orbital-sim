import numpy as np
from constants import R_EARTH
from physics import orbital_velocity, orbital_period, escape_velocity, orbital_energy

#This will get parameters from user
def get_user_input(): 
    print("\n=== Orbital Mechanics Simulator ===\n")