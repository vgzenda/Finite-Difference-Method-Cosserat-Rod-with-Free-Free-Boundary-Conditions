import numpy as np
import matplotlib.pyplot as plt
from cosserat_rod import Cosserat_Rod
from visualize import *



length = 1.0
num_steps = 50  # Number of spatial points
gravity_flag = False
rod = Cosserat_Rod(length=length, num_steps=num_steps,gravity_flag=gravity_flag)

# Update BDF coefficients after setting dt
rod.dt = 0.01      # Time step
rod.N_steps =100  # Number of time steps

# Initial conditions
F_minus = np.zeros(6)  # wrench at base (often 0)
F_plus = np.zeros(6)   # wrench at tip (free tip)
# F_minus[4] = 2*9.81
# F_plus[4] = -3*9.81
# Example: impulse at t=0.1
# F_plus[5] = 0.5  

# toss flag: sets a predefined force at the boundary of the tip
toss_flag = True
rod.toss_flag  =toss_flag

g_SE_list, mcE_list, mcV_list, Lambda_list, nu0_list = rod.time_integrate_free_free(
    F_minus, F_plus, verbose=True
)


np.savez("simulation_results_toss1s.npz", 
         g_SE_list=np.array(g_SE_list), 
         mcE_list=np.array(mcE_list), 
         mcV_list=np.array(mcV_list), 
         nu0_list=np.array(nu0_list), 
         Lambda_list=np.array(Lambda_list), 
         tspan=np.array(rod.tspan)
    )


data = np.load("simulation_results_toss1s.npz")
g_SE_list = data["g_SE_list"]
mcE_list = data["mcE_list"]
mcV_list = data["mcV_list"]
Lambda_list = data["Lambda_list"]
tspan = data["tspan"]


# # Animate results
mcV_list_zeros = np.zeros_like(mcE_list)
fig, anim = rod.animate_cosserat_rod(g_SE_list, mcE_list, mcV_list_zeros, Lambda_list)
plt.show()

# Analyze dynamics (optional, for additional insights)
fig = rod.analyze_rod_dynamics(g_SE_list, mcE_list, mcV_list)
# anim.save('rod_animation_toss5s.mp4', writer='ffmpeg', fps=30, dpi=150)
plt.show()

