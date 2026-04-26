from ik_visualization import StewartPlatform33
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def spiral_trajectory(center, radius, num_frames = 100):
    center = np.array(center)
    positions = []
    orientations = []

    rot_count = 6
    rot_mod = rot_count * 2

    t = np.linspace(0, 1, num_frames)
    phi_max = np.deg2rad(30)
    phi = phi_max * t
    theta = rot_mod * np.pi * t

    xs = radius * np.sin(phi) * np.cos(theta)
    ys = radius * np.sin(phi) * np.sin(theta)
    zs = radius * np.cos(phi)

    points = np.vstack([xs, ys, zs])
    
    Ry_90 = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    points = Ry_90 @ points

    for i in range(num_frames):
        local_p = points[:, i]
        global_p = local_p + center
        positions.append(global_p)

        vec_p_cen = center - global_p
        v = vec_p_cen / np.linalg.norm(vec_p_cen)

        pitch = np.arctan2(np.sqrt(v[0]**2 + v[1]**2), v[2])
        yaw = np.arctan2(v[1], v[0])

        rpy = [0, np.rad2deg(pitch), np.rad2deg(yaw)]
        orientations.append(rpy)

    return positions, orientations

def expand(distance, num_frames = 25):
    start_pos = [0.5, 0, 0]
    positions = []
    orientations = []

    steps = np.linspace(start_pos[0], distance, num_frames)

    for x in steps:
        positions.append([x, 0, 0])
        orientations.append([0, 90, 0])

    return positions, orientations

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
    base_r = 5
    plat_r = base_r

    platform = StewartPlatform33(base_r, plat_r)
    X_init = [1, 0, 0, 0, 90, 0]
    X_goal = [10, 2, 3, 15, 120, 15]

    ani_frame = 100

    # Ensure we get exactly the number of frames expected by FuncAnimation
    path = gen_trajectory(X_init, X_goal, ani_frame)
    
    # Adjust ani_frame to the actual length of the path array
    actual_frames = len(path)
    path_xyz = path[:, :3]
    path_rpy = path[:, 3:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=actual_frames,
        fargs=(platform, path_xyz, path_rpy, ax),
        interval=50
    )

    plt.show()

def update(frame_idx, platform, pos, rpy, ax):
    base_pos = [0, 0, 0]
    base_rpy = [0, 90, 0]

    ax.cla()

    target_pos, target_rpy = pos[frame_idx], rpy[frame_idx]

    _, lines, base_pts, plat_pts = platform.solve_leg_lengths(base_pos, base_rpy, target_pos, target_rpy)

    for name, (start, end) in lines.items():
        ax.plot(
            [start[0], end[0]], 
            [start[1], end[1]], 
            [start[2], end[2]], 
            color='blue', linewidth=2
        )

    platform.plot_triangle(ax, base_pts, ['A', 'B', 'C'], 'black', 'Base')
    platform.plot_triangle(ax, plat_pts, ['D', 'E', 'F'], 'magenta', 'Platform')

    platform.plot_circle(ax, base_pos, platform.base_r, base_rpy, 'black')
    platform.plot_circle(ax, target_pos, platform.plat_r, target_rpy, 'magenta')
    
    base_r = platform.base_r
    vis_r = base_r * 0.8
    platform.plot_cylinder(ax, base_pos, base_rpy, target_pos, target_rpy, vis_r)

    ax.plot(pos[:,0], pos[:,1], pos[:,2], 'g--', alpha=0.3)
    ax.scatter(*target_pos, color='red', s=10)

    ax.set_xlim(-0, 15)
    ax.set_ylim(-7.5, 7.5)
    ax.set_zlim(-7.5, 7.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.axis('Equal')

def main():
    base_r = 5
    plat_r = base_r

    platform = StewartPlatform33(base_r, plat_r)
    X_init = [1, 0, 0, 0, 90, 0]
    X_goal = [10, 2, 3, 15, 120, 15]

    ani_frame = 100

    path = gen_trajectory(X_init, X_goal, ani_frame)
    path_xyz = path[:, :3]
    path_rpy = path[:, 3:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ani = animation.FuncAnimation(
        fig,
        update,
        frames = ani_frame,
        fargs=(platform, path_xyz, path_rpy, ax),
        interval=50
    )

    plt.show()

if __name__ == '__main__':
    main()