"""
Jacobian math from Sam's research
"""
import numpy as np
from inverse_kinematics import StewartPlatform


def jacobian(platform, X=[0, 0, 0, 0, 0, 0]):
    # define connections for base and plat points, leg 1 is point index 1 in base to point index 0 in plat
    leg_con_indices = [(1,0), (2,1), (3,2), (4,3), (5,4), (0,5)]

    _, _, B, _ = platform.solve_ik(platform.X_base, X)  # Bug 1: use X, not X_plat

    t = np.array(X[0:3])
    Rot = platform.rpy2rot(X[3:])
    # Euler-rate: base angle velocity matrix:
    # T = [[cpsi * ct, -spsi, 0,]
    #      [spsi * ct,  cpsi, -;]
    #      [  -st    ,  0,     1]]

    J = np.zeros((6, 6))
    for i, (b_idx, p_idx) in enumerate(leg_con_indices):
        pB = Rot @ platform.local_plat_points[:, p_idx]
        r = t + pB - B[:, b_idx]
        l = np.linalg.norm(r)
        u = r / l
        J[i, 0:3] = u.T
        J[i, 3:] = np.cross(pB, u).T
    return J



def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=120) # more legiable printing

    base_r = 7
    platform_r = 5
    offset_angle = 5
    
    base_pos = [0, 0, 0]
    base_rpy = [0, 90, 0]
    target_pos = [10, 0, 5]
    target_rpy = [5, 95, -5]

    X_base = np.concatenate((base_pos, base_rpy))
    X_target = np.concatenate((target_pos, target_rpy))

    platform = StewartPlatform(base_r, platform_r, offset_angle = offset_angle)
    platform.set_X(X_base, X_target)

    J = jacobian(platform, X_target)

    print(J)



if __name__ == '__main__':
    main()
