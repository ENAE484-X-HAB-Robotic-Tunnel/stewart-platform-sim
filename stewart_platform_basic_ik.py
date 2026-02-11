"""
Derivation from Journal of Mechatronics and Robotics, Inverse Kinematics of a Stewart Platform by R. Petrescu, et al. Oct 2018
and 
The Mathematics of the Stewart Platform by Radamés Ajna and Thiago Hersan as part of memememe for the Object Liberation Front https://olf.alab.space/projects/


Solves for the leg lengths required for the Stewart Platform to reach a given x y z roll pitch yaw
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class StewartPlatform33:
    def __init__(self, r_base, r_plat):
        """
        Initialize the geometry of a 3-3 Stewart Platform
        """

        self.r_base = r_base
        self.r_plat = r_plat

        # Defining local base points, relative to the Base Center
        # Equilaterial Triangle
        # A is at 0 degrees, B is at 120, C is at 240. (Standard CCW order)
        self.local_base_points = {
            'A': np.array([r_base, 0, 0]),
            'B': np.array([-0.5 * r_base, np.sqrt(3)/2 * r_base, 0]),  # Fixed: +Y for 120 deg
            'C': np.array([-0.5 * r_base, -np.sqrt(3)/2 * r_base, 0])  # Fixed: -Y for 240 deg
        }

        # Defining local platform points, relative to platform center
        # D goes to A and B, E goes to B and C, F goes to C and A

        # Top platform is rotated by 60 degrees for 3-3 geometry
        angle = np.deg2rad(60)

        Rz_60 = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])

        self.local_plat_points = {
            'D': Rz_60 @ np.array([r_plat, 0, 0]),
            'E': Rz_60 @ np.array([-0.5 * r_plat, np.sqrt(3)/2 * r_plat, 0]), # Fixed signs here too to match
            'F': Rz_60 @ np.array([-0.5 * r_plat, -np.sqrt(3)/2 * r_plat, 0])
        }

    def get_rpy(self, rpy):
        # obtain rotation matrix given row pitch and yaw
        rpy = np.deg2rad(rpy)
        roll, pitch, yaw = rpy
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])

        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx
    
    def solve_leg_lengths(self, base_pos, target_pos, target_rpy):
        # 321 rotation matrix from base to target frame
        R_plat = self.get_rpy(target_rpy)

        # define how the legs are connected
        leg_con = [
            ('Leg1', 'A', 'D'), # Leg 1 connects Base A to Plat D
            ('Leg2', 'B', 'D'), # Leg 2 connects Base B to Plat D
            ('Leg3', 'B', 'E'), # ...
            ('Leg4', 'C', 'E'),
            ('Leg5', 'C', 'F'),
            ('Leg6', 'A', 'F')
        ]

        lengths = {}
        lines = {}
        
        # New: Dictionaries to store global coordinates for plotting triangles
        base_coords = {}
        plat_coords = {}

        # target position and base position in base frame
        T = np.array(target_pos)
        B = np.array(base_pos)

        # solve for leg lengths
        for leg_name, base_key, plat_key in leg_con:
            # start by obtaining local coordinates p_i and b_i
            b_i = self.local_base_points[base_key]
            p_i = self.local_plat_points[plat_key]

            # Convert to base frame
            base_point = B + b_i
            plat_point = T + (R_plat @ p_i)
            
            # Store for plotting
            base_coords[base_key] = base_point
            plat_coords[plat_key] = plat_point

            # find leg lengths
            leg_vec = plat_point - base_point

            lengths[leg_name] = np.linalg.norm(leg_vec)
            lines[leg_name] = (base_point, plat_point)

        return lengths, lines, base_coords, plat_coords


if __name__ == '__main__':

    # give the target coords and rpy in deg

    platform = StewartPlatform33(15, 15) # give base and platform radius

    # x y z
    base_pos = [0, 0, 0]
    target_pos = [0, 0, 50]
    target_rpy = [0, 0, 0] # deg

    lengths, lines, base_pts, plat_pts = platform.solve_leg_lengths(base_pos, target_pos, target_rpy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot legs
    for name, (start, end) in lines.items():
        ax.plot(
            [start[0], end[0]], # x
            [start[1], end[1]], # y
            [start[2], end[2]], # z
            color='blue', linewidth=2
        )

    # plotting base and platform triangles showing the connection points
    def plot_triangle(points, keys, color, label):
        # draw a triangle from going to a -> b -> c -> a
        x = [points[k][0] for k in keys] + [points[keys[0]][0]] # A_x -> B_x -> C_x -> A_x
        y = [points[k][1] for k in keys] + [points[keys[0]][1]] # repeat for y
        z = [points[k][2] for k in keys] + [points[keys[0]][2]] # repeat for z
        ax.plot(x, y, z, color=color, label=label)

    plot_triangle(base_pts, ['A', 'B', 'C'], 'black', 'Base')
    plot_triangle(plat_pts, ['D', 'E', 'F'], 'magenta', 'Platform')

    # plot platforms
    def plot_circle(center, r, rpy=[0,0,0], color='black'):
        theta = np.linspace(0, 2*np.pi, 100)
        # Create circle in local XY plane
        xc = r * np.cos(theta) # x coord of circle
        yc = r * np.sin(theta) # y
        zc = np.zeros_like(xc) # z
        
        # apply rotation and translation
        points = np.vstack([xc, yc, zc])
        R = platform.get_rpy(rpy) # get rotation matrix
        points = R @ points + np.array(center).reshape(3, 1)
        
        ax.plot(points[0,:], points[1,:], points[2,:], color=color, linestyle='--')

    plot_circle(base_pos, platform.r_base, [0,0,0], 'black')
    plot_circle(target_pos, platform.r_plat, target_rpy, 'magenta')

    # calculate the Normal Vector
    R_current = platform.get_rpy(target_rpy)
    platform_normal = np.array([0, 0, 1]) * 10 # want the normal to be 10 'units' away

    # transform normal from platform frame to base frame
    base_frame_normal = R_current @ platform_normal\
    
    # define start and end position of normal vector
    start = np.array(target_pos)
    end = start + base_frame_normal
    
    ax.quiver(
            target_pos[0], target_pos[1], target_pos[2],
            base_frame_normal[0], base_frame_normal[1], base_frame_normal[2],
            length=1,
            color='green', 
            linewidth=2,
            arrow_length_ratio=0.1,
            label='Normal Vector'
        )
    
    ax.legend()
    plt.axis('equal')
    plt.show()