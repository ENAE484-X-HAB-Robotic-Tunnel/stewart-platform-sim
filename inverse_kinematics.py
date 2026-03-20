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
        
        # half of the offset angle
        phi = np.deg2rad(offset_angle) / 2

        # local base points (point pairs are centered at 0, 120, 240 deg)
        self.local_base_points = {
            'B1': np.array([base_r * np.cos(0 - phi),   base_r * np.sin(0 - phi),   0]),
            'B2': np.array([base_r * np.cos(0 + phi),   base_r * np.sin(0 + phi),   0]),
            'B3': np.array([base_r * np.cos(np.deg2rad(120) - phi), base_r * np.sin(np.deg2rad(120) - phi), 0]),
            'B4': np.array([base_r * np.cos(np.deg2rad(120) + phi), base_r * np.sin(np.deg2rad(120) + phi), 0]),
            'B5': np.array([base_r * np.cos(np.deg2rad(240) - phi), base_r * np.sin(np.deg2rad(240) - phi), 0]),
            'B6': np.array([base_r * np.cos(np.deg2rad(240) + phi), base_r * np.sin(np.deg2rad(240) + phi), 0])
        }

        # local platform points (point pairs are centered at 60, 180, 300 deg)
        self.local_plat_points = {
            'P1': np.array([plat_r * np.cos(np.deg2rad(60) - phi),  plat_r * np.sin(np.deg2rad(60) - phi),  0]),
            'P2': np.array([plat_r * np.cos(np.deg2rad(60) + phi),  plat_r * np.sin(np.deg2rad(60) + phi),  0]),
            'P3': np.array([plat_r * np.cos(np.deg2rad(180) - phi), plat_r * np.sin(np.deg2rad(180) - phi), 0]),
            'P4': np.array([plat_r * np.cos(np.deg2rad(180) + phi), plat_r * np.sin(np.deg2rad(180) + phi), 0]),
            'P5': np.array([plat_r * np.cos(np.deg2rad(300) - phi), plat_r * np.sin(np.deg2rad(300) - phi), 0]),
            'P6': np.array([plat_r * np.cos(np.deg2rad(300) + phi), plat_r * np.sin(np.deg2rad(300) + phi), 0])
        }

    def get_rpy(self, rpy):
        rpy = np.deg2rad(rpy)
        roll, pitch, yaw = rpy
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def solve_leg_lengths(self, base_pos, base_rpy, target_pos, target_rpy):
        base_r_mat = self.get_rpy(base_rpy)
        plat_r_mat = self.get_rpy(target_rpy)

        # 6-6 Leg connectivity
        leg_con = [
            ('Leg1', 'B2', 'P1'),
            ('Leg2', 'B3', 'P2'),
            ('Leg3', 'B4', 'P3'),
            ('Leg4', 'B5', 'P4'),
            ('Leg5', 'B6', 'P5'),
            ('Leg6', 'B1', 'P6')
        ]

        lengths, lines, base_coords, plat_coords = {}, {}, {}, {}
        T = np.array(target_pos) - np.array(base_pos)
        B = np.array(base_pos)

        for leg_name, b_key, p_key in leg_con:
            b_i, p_i = self.local_base_points[b_key], self.local_plat_points[p_key]
            base_point = B + (base_r_mat @ b_i)
            plat_point = T + (plat_r_mat @ p_i)
            base_coords[b_key], plat_coords[p_key] = base_point, plat_point
            leg_vec = plat_point - base_point
            lengths[leg_name] = np.linalg.norm(leg_vec)
            lines[leg_name] = (base_point, plat_point)

        return lengths, lines, base_coords, plat_coords

    def plot_triangle(self, ax, points, keys, color, label):
        x = [points[k][0] for k in keys] + [points[keys[0]][0]]
        y = [points[k][1] for k in keys] + [points[keys[0]][1]]
        z = [points[k][2] for k in keys] + [points[keys[0]][2]]
        ax.plot(x, y, z, color=color, label=label)

    def plot_circle(self, ax, center, r, rpy=[0,0,0], color='black'):
        theta = np.linspace(0, 2*np.pi, 100)
        points = np.vstack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)])
        R = self.get_rpy(rpy)
        points = R @ points + np.array(center).reshape(3, 1)
        ax.plot(points[0,:], points[1,:], points[2,:], color=color, linestyle='--')

    def plot_normal(self, ax, target_pos, target_rpy):
        plat_r = self.get_rpy(target_rpy)
        base_frame_normal = plat_r @ np.array([0, 0, 1])
        ax.quiver(target_pos[0], target_pos[1], target_pos[2], base_frame_normal[0], base_frame_normal[1], base_frame_normal[2], length=1, color='green')

def main():
    base_r = 7
    platform_r = 5
    offset_angle = 5 # offset angle between close points

    # no offset angle causes platform to refert to 33
    platform = StewartPlatform(base_r, platform_r, offset_angle = offset_angle)

    base_pos = [0, 0, 0]
    base_rpy = [0, 90, 0]
    target_pos = [10, 0, 5]
    target_rpy = [5, 95, -5]

    lengths, lines, base_pts, plat_pts = platform.solve_leg_lengths(base_pos, base_rpy, target_pos, target_rpy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for name, (start, end) in lines.items():
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=2)

    # plotting keys
    base_keys = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
    plat_keys = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    platform.plot_triangle(ax, base_pts, base_keys, 'black', 'Base')
    platform.plot_triangle(ax, plat_pts, plat_keys, 'magenta', 'Platform')

    platform.plot_circle(ax, base_pos, platform.base_r, base_rpy, 'black')
    platform.plot_circle(ax, target_pos, platform.plat_r, target_rpy, 'magenta')
    platform.plot_normal(ax, target_pos, target_rpy)

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()