"""

Newton-Raphson iterative FK method
1 d = IK(p, R)
2 J = Jac(p, R)
3 d x_ = J.inv()(d_goal - d_)
4 p = p + dp
  R = dR * R

5 IF ||d_goal - d|| < tolerance

else go to 1
"""

import numpy as np
from inverse_kinematics import StewartPlatform
from jacobian import jacobian

def forward_kinematics(platform, X_prev, d_goal):
    """
    Inputs:
    X_prev - xyz rpy of the previous state
    
    d_goal - ik goal lengths
    
    Output:
    X - Approximated state
    """
    tolerance = 0.001
    X_current = np.array(X_prev)

    for _ in range(1000):
        p = X_current[:3]
        rpy = X_current[3:]

        d_guess, _, _, _ = platform.solve_ik(platform.X_base, X_current)

        if np.linalg.norm(d_goal - d_guess) < tolerance:
            break

        J = jacobian(platform, X_current)
        
        alpha = 0.5  # damping factor to prevent over rotation from during rot update
        dx = alpha * np.linalg.pinv(J, rcond=1e-4) @ (d_goal - d_guess)
        
        dp = dx[:3]
        drpy = dx[3:]
        
        p = p + dp

        Rot = platform.rpy2rot(rpy)
        dRot = platform.rpy2rot(drpy, degree=False)

        Rot = dRot @ Rot

        rpy = platform.rot2rpy(Rot)
        
        X_current = np.concatenate((p, rpy))

    # rotate back to global frame
    X_current[3:] = platform.post_rotate(X_current[3:])
    return X_current

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    base_r = 7
    platform_r = 5
    offset_angle = 5 
    platform = StewartPlatform(base_r, platform_r, offset_angle=offset_angle)

    # prerotate global frame so that yaw = 0 to prevent gimbal locking -> yaw at 90 induces gimbal lock
    base_pos = [0, 0, 0]
    base_rpy = platform.pre_rotate([0, 90, 0])
    X_base = np.concatenate((base_pos, base_rpy))

    goal_pos = [10, 0, 2]
    goal_rpy = platform.pre_rotate([5, 90, 25])
    X_goal = np.concatenate((goal_pos, goal_rpy))
    
    prev_pos = [1, 0, 0]
    prev_rpy = platform.pre_rotate([2, 90, 0])
    X_prev = np.concatenate((prev_pos, prev_rpy))




    platform.set_X(X_base)

    d_goal, _, _, _ = platform.solve_ik(X_base, X_goal)

    print(forward_kinematics(platform, X_prev, d_goal))

if __name__ == '__main__':
    main()
