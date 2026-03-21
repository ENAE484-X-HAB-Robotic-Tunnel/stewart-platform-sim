import numpy as np
import matplotlib.pyplot as plt
import time

class StewartPlatform:
    def __init__(self, base_r, plat_r, offset_angle=0):
        """
        Initialize the geometry of a 6-6 Stewart Platform
        offset_angle: angle in degrees between two close points
        """
        self.base_r = base_r
        self.plat_r = plat_r

        self.X_base = np.zeros(6)
        self.X_plat = np.zeros(6)
        
        phi = np.deg2rad(offset_angle) / 2

        base_angles = np.array([0 - phi, 0 + phi, np.deg2rad(120) - phi, np.deg2rad(120) + phi, np.deg2rad(240) - phi, np.deg2rad(240) + phi])
        self.local_base_points = np.zeros((3, 6))
        self.local_base_points[0, :] = base_r * np.cos(base_angles)
        self.local_base_points[1, :] = base_r * np.sin(base_angles)

        plat_angles = np.array([np.deg2rad(60) - phi, np.deg2rad(60) + phi, np.deg2rad(180) - phi, np.deg2rad(180) + phi, np.deg2rad(300) - phi, np.deg2rad(300) + phi])
        self.local_plat_points = np.zeros((3, 6))
        self.local_plat_points[0, :] = plat_r * np.cos(plat_angles)
        self.local_plat_points[1, :] = plat_r * np.sin(plat_angles)

    def set_X(self, X_base = [0, 0, 0, 0, 90, 0], X_plat = [1, 0, 0, 0, 90, 0]):
        self.X_base = X_base
        self.X_plat = X_plat

    def pre_rotate(self, rpy):
        pitch_offset = -90
        r, p, y = rpy
        return (r, p + pitch_offset, y)

    def post_rotate(self, rpy):
        pitch_offset = 90
        r, p, y = rpy
        return (r, p + pitch_offset, y)

    def rpy2rot(self, rpy, degree = True):
        if degree == True:
            rpy = np.deg2rad(rpy)
        roll, pitch, yaw = rpy
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx
    
    def rot2rpy(self, R, degree=True):
        # check for gimbal lock
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll  = np.arctan2( R[2, 1],  R[2, 2])
            pitch = np.arctan2(-R[2, 0],  sy)
            yaw   = np.arctan2( R[1, 0],  R[0, 0])
        else:
            pitch = np.arctan2(-R[2, 0], sy)  # still ±pi/2
            if R[2, 0] < 0:  # pitch = +90
                roll = 0.0
                yaw  = np.arctan2(R[0, 1], R[1, 1])
            else:             # pitch = -90
                roll = 0.0
                yaw  = -np.arctan2(R[0, 1], R[1, 1])

        rpy = np.array([roll, pitch, yaw])
        return np.rad2deg(rpy) if degree else rpy


    def solve_ik(self, X_base, X_plat):
        """
        Output:
        lengths:         length of a leg
        lines:           start and end point of a leg
        base_coords:     point pos on base
        plat_coords:     point pos on plat
        """
        base_pos = X_base[:3]
        base_rpy = X_base[3:]
        target_pos = X_plat[:3]
        target_rpy = X_plat[3:]

        base_r_mat = self.rpy2rot(base_rpy)
        plat_r_mat = self.rpy2rot(target_rpy)

        leg_con_indices = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (0, 5)]

        T = (np.array(target_pos) - np.array(base_pos)).reshape(3, 1)
        B = np.array(base_pos).reshape(3, 1)

        base_coords = B + (base_r_mat @ self.local_base_points)
        plat_coords = T + (plat_r_mat @ self.local_plat_points)

        lengths = np.zeros(6)
        lines = np.zeros((6, 2, 3)) 

        for i, (b_idx, p_idx) in enumerate(leg_con_indices):
            base_point = base_coords[:, b_idx]
            plat_point = plat_coords[:, p_idx]
            leg_vec = plat_point - base_point
            lengths[i] = np.linalg.norm(leg_vec)
            lines[i] = (base_point, plat_point)

        return lengths, lines, base_coords, plat_coords

    def plot_triangle(self, ax, points, color, label):
        x = list(points[0, :]) + [points[0, 0]]
        y = list(points[1, :]) + [points[1, 0]]
        z = list(points[2, :]) + [points[2, 0]]
        ax.plot(x, y, z, color=color, label=label)

    def plot_circle(self, ax, center, r, rpy=[0,0,0], color='black'):
        theta = np.linspace(0, 2*np.pi, 100)
        points = np.vstack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)])
        R = self.rpy2rot(rpy)
        points = R @ points + np.array(center).reshape(3, 1)
        ax.plot(points[0,:], points[1,:], points[2,:], color=color, linestyle='--')

    def plot_normal(self, ax, target_pos, target_rpy):
        plat_r = self.rpy2rot(target_rpy)
        base_frame_normal = plat_r @ np.array([0, 0, 1])
        ax.quiver(target_pos[0], target_pos[1], target_pos[2], base_frame_normal[0], base_frame_normal[1], base_frame_normal[2], length=1, color='green')

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    base_r = 7
    platform_r = 5
    offset_angle = 5

    platform = StewartPlatform(base_r, platform_r, offset_angle = offset_angle)

    base_pos = [0, 0, 0]
    base_rpy = [0, 90, 0]
    X_base = np.concatenate((base_pos, base_rpy))
    goal_pos = [2, 0, 0]
    goal_rpy = [5, 90, 0]
    X_goal = np.concatenate((goal_pos, goal_rpy))

    lengths, lines, base_pts, plat_pts = platform.solve_ik(X_base, X_goal)
    # leg_con_indices = [(1,0), (2,1), (3,2), (4,3), (5,4), (0,5)]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for leg_pts in lines:
        start, end = leg_pts[0], leg_pts[1]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=2)

    platform.plot_triangle(ax, base_pts, 'black', 'Base')
    platform.plot_triangle(ax, plat_pts, 'magenta', 'Platform')

    platform.plot_circle(ax, X_base[:3], platform.base_r, X_base[3:], 'black')
    platform.plot_circle(ax, X_goal[:3], platform.plat_r, X_goal[3:], 'magenta')
    platform.plot_normal(ax, X_goal[:3], X_goal[3:])

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()