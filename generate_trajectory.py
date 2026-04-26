import numpy as np


def gen_trajectory(X_init, X_goal, N=100):
    """
    Two phases, pre-dock and docking

    Pre-Dock: Reach target y, z, r, p, y when x = x_goal - spacing

    Docking: Vary x until x = x_goal
    """
    X_init = np.array(X_init, dtype=float)
    X_goal = np.array(X_goal, dtype=float)
    
    spacing = X_goal[0] * 0.25
    x_split = X_goal[0] - spacing
    total_x_dist = X_goal[0] - X_init[0]

    # allocate steps to each phase, proportional to size of dock x vs pre-dock x
    phase_2_ratio = spacing / total_x_dist
    n_phase2 = max(1, int(N * phase_2_ratio)) 
    n_phase1 = N - n_phase2

    path = []

    # phase 1: move to pre-dock pose. 
    # y, z, r, p, y, reach goal values
    # get start and end states for phase 1
    start_state_p1 = X_init.copy()
    end_state_p1 = X_goal.copy()
    end_state_p1[0] = x_split

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

    