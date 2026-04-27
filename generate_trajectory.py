import numpy as np

def rpy2rot(rpy, degree = False):
    if degree == True:
        rpy = np.deg2rad(rpy)
    roll, pitch, yaw = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def gen_trajectory(X_init, X_goal, N=100):
    """
    Two phases, pre-dock and docking

    Pre-Dock: reach spacing normal distance away from the docking hatch

    Docking: approach the dock varying normal distance only
    """
    X_init = np.array(X_init) # poses into numpy arrays
    X_goal = np.array(X_goal)
    spacing = 3

    rpy = np.deg2rad(X_goal[3:]) # convert rpy from deg to radians

    R_goal = rpy2rot(rpy)


    split_point = R_goal @ np.array([0, 0, spacing])
    split_point = X_goal[:3] - split_point

    phase_1_length = np.linalg.norm((split_point - X_init[:3]))
    phase_2_length = np.linalg.norm((split_point - X_goal[:3]))
    total_length = phase_1_length + phase_2_length

    phase_2_ratio = phase_2_length / total_length
    n_phase2 = max(1, int(N*phase_2_ratio))
    n_phase1 = N - n_phase2
    

    path = []

    """
    extend normal vector off of docking pose, up to spacing. Then draw trajectory
    from initial position to the cutoff point
    
    """

    # phase 1: move to pre-dock pose. 
    # y, z, r, p, y, reach goal values
    # get start and end states for phase 1
    start_state_p1 = X_init.copy()
    end_state_p1 = np.zeros(6)
    end_state_p1[:3] = split_point
    end_state_p1[3:] = np.rad2deg(rpy)

    # populate path with stespsteps
    t1 = np.linspace(0, 1, n_phase1)
    for frac in t1:
        p1_state = start_state_p1 + (end_state_p1 - start_state_p1) * frac
        path.append(p1_state)

    # phase 2: move from pre-dock to dock
    # x goes from X_goal - spacing to X_goal
    # get start and end states for phase 2
    start_state_p2 = end_state_p1.copy()
    end_state_p2 = X_goal.copy()

    # populate path with steps
    t2 = np.linspace(0, 1, n_phase2 + 1) 
    for frac in t2[1:]: # avoid second step where phase 1 ends
        p2_state = start_state_p2 + (end_state_p2 - start_state_p2) * frac
        path.append(p2_state)

    return np.array(path)

def main():
    X_init = [1, 0, 0, 0, 90, 0]
    X_goal = [5, 2, 2, 45, 90, 45]

if __name__ == '__main__':
    main()

    