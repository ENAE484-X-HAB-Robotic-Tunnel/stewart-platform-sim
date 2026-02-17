from ik_visualization import StewartPlatform33

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def trajectory(center, radius, num_frames = 100):
    # creates a trajectory of a spherical cap
    # take the top then rotate it 90 deg about y

    center = np.array(center)

    # initialize arrays to store pos and orientation goal of platform
    positions = []
    orientations = []

    rot_count = 6
    rot_mod = rot_count * 2

    # create points for path
    t = np.linspace(0, 1, num_frames)
    phi_max = np.deg2rad(30)
    phi = phi_max * t
    theta = rot_mod * np.pi * t

    xs = radius * np.sin(phi) * np.cos(theta)
    ys = radius * np.sin(phi) * np.sin(theta)
    zs = radius * np.cos(phi)

    points = np.vstack([xs, ys, zs])
    
    # align with tunnel
    Ry_90 = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    points = Ry_90 @ points

    for i in range(num_frames):
        # translate points to 'center' of sphere
        local_p = points[:, i]
        global_p = local_p + center
        positions.append(global_p)

        # calculate orientaion for platform to look at the center of the sphere
        vec_p_cen = center - global_p
        v = vec_p_cen / np.linalg.norm(vec_p_cen)

        # obtain rpy, assume no roll
        pitch = np.arctan2(np.sqrt(v[0]**2 + v[1]**2), v[2])
        yaw = np.arctan2(v[1], v[0])

        rpy = [0, np.rad2deg(pitch), np.rad2deg(yaw)]
        orientations.append(rpy)

    return positions, orientations

def update(frame_idx, platform, pos, rpy, ax):
    """
    Handles updating the animation
    1) Clear plot
    2) obtain leg lengths by solving ik
    3) plot new platform
    4) plot the trajectory
    5) handle labels and stuff
    """

    # initalize platform params
    base_pos = [0, 0, 0]
    base_rpy = [0, 90, 0]

    # clear
    ax.cla()

    # get target for this frame
    pos = np.array(pos)
    rpy = np.array(rpy)
    target_pos, target_rpy = pos[frame_idx], rpy[frame_idx]

    # ik
    _, lines, base_pts, plat_pts = platform.solve_leg_lengths(base_pos, base_rpy, target_pos, target_rpy)

    # drawing 

    # plot legs
    # manage connectivity, start and end position of each line in x, y, and z
    for name, (start, end) in lines.items():
        ax.plot(
            [start[0], end[0]], # x
            [start[1], end[1]], # y
            [start[2], end[2]], # z
            color='blue', linewidth=2
        )

    # plot triangle connecting legs on each platform
    platform.plot_triangle(ax, base_pts, ['A', 'B', 'C'], 'black', 'Base')
    platform.plot_triangle(ax, plat_pts, ['D', 'E', 'F'], 'magenta', 'Platform')

    # plot platforms
    platform.plot_circle(ax, base_pos, platform.base_r, base_rpy, 'black')
    platform.plot_circle(ax, target_pos, platform.plat_r, target_rpy, 'magenta')

    # plot the Normal Vector
    # platform.plot_normal(ax, target_pos, target_rpy)
    # doesn't seem to work for the animation

    
    # plot cylinder to visualize tunnel
    base_r = platform.base_r
    vis_r = base_r * 0.8
    platform.plot_cylinder(ax, base_pos, base_rpy, target_pos, target_rpy, vis_r)


    # plot trajectory
    ax.plot(pos[:,0], pos[:,1], pos[:,2], 'g--', alpha=0.3)
    # plot current target as a dot
    ax.scatter(*target_pos, color='red', s=10)

    # axis and stuff
    ax.legend()
    ax.set_xlim(-0, 15)
    ax.set_ylim(-7.5, 7.5)
    ax.set_zlim(-7.5, 7.5)


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.axis('equal')



def main():
    # platform initialization
    base_r = 5
    plat_r = base_r

    platform = StewartPlatform33(base_r, plat_r)

    num_frames = 500

    trag_cen = [25, 0, 0]
    trag_r = 15
    path, path_rpy = trajectory(trag_cen, trag_r, num_frames)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # create animation object
    ani = animation.FuncAnimation(
        fig,
        update,
        frames = num_frames,
        fargs=(platform, path, path_rpy, ax),
        interval=50
    )

    plt.show()

    # ani.save('stewart.gif', writer='pillow', fps=20)
    # this is broken for some reason




if __name__ == '__main__':
    main()