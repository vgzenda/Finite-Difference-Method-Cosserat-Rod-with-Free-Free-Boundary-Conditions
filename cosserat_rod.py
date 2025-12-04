# Vaughn Gzenda 
# 2025

import numpy as np
import matplotlib.pyplot as plt 
from liegroups import * 
from visualize import * 
from scipy.optimize import minimize, least_squares

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D



class Cosserat_Rod:
    def __init__(self,length,num_steps,gravity_flag=False):
        # Parameters 
        self.L = length                          # length
        self.N = num_steps                      # number of elements (spatial discretization)
        self.s_points = np.linspace(0,self.L,self.N)
        print('self.s_points',self.s_points[-1])
        print('len(s_points)',len(self.s_points))
        self.dt = 0.0025                          # time step for BDF2
        self.F_tip = np.zeros(3)                 # tip force is zero
        self.M_tip = np.zeros(3)                 # tip moment is zero
        self.c_0 = 1.5/self.dt
        self.c_1 = -2/self.dt 
        self.c_2 = 0.5/self.dt
        self.N_steps = 10

        self.t0 = 0.0
        self.integration_type = 'RK4'
        # self.integration_type = 'Euler'
        
        # Boundary Conditions 
        self.g0 = np.eye(4)                       # initial config
        self.mcV0 = np.array([0, 0, 0, 0, 0, 0])  # initial velocity
        self.mcE0 = np.array([0, 0, 0, 0, 0, 0])  # initial velocity
        self.target_Lambda_tip = np.zeros((6,))

        self.gravity_flag = gravity_flag
        self.toss_flag = False

        # initialize base 
        self.init_base_history(R0=np.eye(3), r0=np.zeros(3),
                       Omega0=np.zeros(3), Omegadot0=np.zeros(3),
                       rdot0=np.zeros(3), rddot0=np.zeros(3))


    def update_bdf_coefficients(self):
        self.c_0 = 1.5 / self.dt
        self.c_1 = -2 / self.dt
        self.c_2 = 0.5 / self.dt



    # free free inital conditions 
    def init_base_history(self, R0=None, r0=None, Omega0=None, Omegadot0=None, rdot0=None, rddot0=None):
        """Initialize base history used by Newmark implicit integrator.
        Provide reasonable defaults if None (zero motion).
        """
        if R0 is None:
            R0 = np.eye(3)
        if r0 is None:
            r0 = np.zeros(3)
        if Omega0 is None:
            Omega0 = np.zeros(3)
        if Omegadot0 is None:
            Omegadot0 = np.zeros(3)
        if rdot0 is None:
            rdot0 = np.zeros(3)
        if rddot0 is None:
            rddot0 = np.zeros(3)

        # store the previous-step newmark state
        self.R0_n = R0.copy()
        self.r0_n = r0.copy()
        self.Omega0_n = Omega0.copy()
        self.Omegadot0_n = Omegadot0.copy()
        self.rdot0_n = rdot0.copy()
        self.rddot0_n = rddot0.copy()

        # newmark params 
        self.beta = 0.25
        self.gamma = 0.5

        # critcal time step 
        self.soft_robot_parameters()
        self.critical_time_increment()



    def soft_robot_parameters(self):

        # self.rho = 75e1                 # [kg/m3]    Density of Material
        self.mu = 0.0                     # [N/m^2s]   Viscosity of Peanut Butter
        # self.r = 0.01                   # [m]        Radius of Cross-Section
        # self.E = 5e8                    # [Pa]       Youngs Modulus
        # G = self.E / (2 * (1 + 0.3))    # [Pa]       Shear Modulus
        self.rho = 1e2
        self.r = 0.1
        self.E = 5e6                      # [Pa]       Youngs Modulus
        nu = 0.3                         # [dimless]  Poisson ratio 
        G = self.E/(2*(1+nu))            # [Pa]       Shear Modulus
        A = np.pi * self.r**2            # [m2]       Cross-Sectional Area of Beam
        I = np.pi / 4 * self.r**4        # [m4]       2nd Moment of Inertia of Beam
        self.I = I

        # Calculate J, Kbt, Kse, Cse, Cbt 
        # print('I:', I)
        J = np.diag([2 * I, I, I])            # [m4]       3D Moment of Inertia of Beam
        Kbt = np.diag([2 * G * I, self.E * I, self.E * I])    # [Nm^2]     Bending and Torsional Rigidity (Rotational)
        Kse = np.diag([self.E * A, G * A, G * A])      # [N]        Shear and Extension Rigidity (Linear)
        Cse = np.diag([3 * A, A, A]) * self.mu       # [N/s]      Shear and Extension Damping (Linear)
        Cbt = np.diag([2 * I, I, I]) * self.mu       # [N/s]      Bending and Torsional Damping (Rotational)

        m_linear = self.rho * A * np.eye(3)
        m_rotational = self.rho * J
        # print('det J', np.linalg.det(J))
        # print('det m_rotational', np.linalg.det(m_rotational))
        # print('det m_linear', np.linalg.det(m_linear))
        # mass matrix
        mbM = np.block([[m_rotational, np.zeros((3, 3))],
                        [np.zeros((3, 3)), m_linear]])
        # stiffness matrix
        mbK = np.block([[Kbt, np.zeros((3, 3))],
                        [np.zeros((3, 3)), Kse]])
        # damping matrix
        mbD = np.block([[Cbt, np.zeros((3, 3))],
                        [np.zeros((3, 3)), Cse]])
        return mbM, mbK, mbD
    
    def critical_time_increment(self):
        """approximates the critical time step (modulo the dimensionless factor chi)"""
        A = np.pi*self.r**2
        top = self.rho*A
        bottom = self.E*self.I
        Deltat = self.L**2*np.sqrt(top/bottom)
        print(f'Critital time step {Deltat}')
        self.Deltat_c = Deltat

    
    # ================================================================== Equations of motion
    def actuation_wrench(self,s,mcE):
        Lambda_a = np.zeros((6,))
        return Lambda_a

    def gravity_wrench(self,g_SE,mbM):
        """Computes the gravity wrench F(s,t)."""
        if self.gravity_flag:
            G = np.array([0, 0, 0, 0, 0, -9.81])
        else:
            G = np.zeros((6,))
        # G = np.zeros((6,))
        F_g = mbM@Adjoint_SE3(SE3_inv(g_SE))@G
        return F_g.reshape((6,))

    def mcE_rest(self,s):
        """Rest strain field mcE(s) for the Cosserat rod."""
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    #     return np.array([0.0, 0.0, 0.0,  np.sin(s), 0.0 , 0.0])

    def mcE_star_vec(self):
        mcE_star = np.zeros((self.N,6))
        for i in range(self.N):
            mcE_star[i,:] = self.mcE_rest(self.s_points[i])
        self.mcE_star = mcE_star
        return mcE_star

    def constitutive_law(self,s,mcE,Lambda_a,mbK,mbD,mcE_t):
        # rest strain 
        mcE_star = self.mcE_rest(s)
    #     print('Lambda_a',Lambda_a)
        # linear elastic constitucive law iwth active strain
        Lambda = mbK@(mcE - mcE_star) + Lambda_a + mbD@mcE_t
    #     print('Lambda',Lambda)
        return Lambda

    def get_strain(self, s, Lambda, Lambda_a, mbK, mbD, mcE_h):
        mcE_star = self.mcE_rest(s)
        mat = mbK + self.c_0 * mbD
        rhs = Lambda - Lambda_a + mbK @ mcE_star - mbD @ mcE_h
        mcE = np.linalg.solve(mat, rhs)
        return mcE

    def semi_discretized_cosserat_equations(self,s, mcE, mcV, mcE_h, mcV_h,mbM,mbK,mbD,F_ext,Lambda_a):
        """Computes the semi-discretized Cosserat rod equations."""
        mcV_t = self.c_0*mcV + mcV_h  # velocity field at time t
        mcE_t = self.c_0*mcE + mcE_h  # strain field at time t
        # velocity field
        mcV_prime = mcE_t + adjoint_SE3(mcV)@mcE
        # strain field
        Lambda = self.constitutive_law(s,mcE,Lambda_a,mbK,mbD,mcE_t)
        Lambda_prime = mbM@mcV_t - coadjoint_SE3(mcV)@mbM@mcV + coadjoint_SE3(mcE)@Lambda  - F_ext
        
        return Lambda_prime, mcV_prime

    def integrate_rod(self,Lambda0, g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev,mcV_pprev, s_points):
        """Integrates the semi-discretized Cosserat rod equations. in space"""
        # update the BDF coeffients
        self.update_bdf_coefficients()
        # allocate memory for the states
        Lambda = np.zeros((self.N, 6))  # stress field
        mcV = np.zeros((self.N, 6))  # velocity field
        g_SE = np.zeros((self.N, 4, 4))  # SE(3) configuration
        mcE = np.zeros((self.N, 6))  # strain field 
        Lambda_prime = np.zeros((self.N, 6))  # velocity field
        mcV_prime = np.zeros((self.N, 6))  # velocity field
        
        
        # set the history vectors from the previous time step
        mcE_h = self.c_1*mcE_prev + self.c_2*mcE_pprev
        mcV_h = self.c_1*mcV_prev + self.c_2*mcV_pprev

        # set initial conditions 
        Lambda[0] = Lambda0
        mcV[0] = mcV0
        g_SE[0] = g_SE0
        # set initial conditions for strain 
        mbM, mbK,mbD = self.soft_robot_parameters()
         # actuator
        Lambda_a = np.zeros((6,))
        mcE[0] = self.get_strain(s_points[0],Lambda[0],Lambda_a,mbK,mbD,mcE_h[0])
       
        if self.integration_type == 'Euler':
            # # integration loop 
            for k in range(1,self.N):
                # print("Before")
                # visualize_frame(g_SE[k-1],mcV=mcV[k-1],mcE=mcE[k-1])
                s = s_points[k-1]  # current arc length parameter
                # print('s:',s)
                # step size
                ds = s_points[k] - s_points[k-1]  # step size in arc length
                # external loads 
                F_ext = self.gravity_wrench(g_SE[k-1],mbM) 
                # actuator inputs 
                Lambda_a = self.actuation_wrench(s, mcE[k-1])
                # print(f"s={s:.3f}, Lambda_a = {Lambda_a}")
                # Compute dynamics (Cosserat equations)
                Lambda_prime_vf, mcV_prime_vf = self.semi_discretized_cosserat_equations(s,mcE[k-1],mcV[k-1],mcE_h[k-1],mcV_h[k-1],mbM,mbK,mbD,F_ext,Lambda_a)
                # Euler spatial integration
                mcV[k] = mcV[k-1] + ds * mcV_prime_vf
                Lambda[k] = Lambda[k-1] + ds * Lambda_prime_vf
                # compute strain 
                # mcE[k] = self.get_strain(s,Lambda[k],Lambda_a,mbK)
                mcE[k] = self.get_strain(s,Lambda[k],Lambda_a,mbK,mbD,mcE_h[k])
                g_SE[k] = g_SE[k-1] @ exp_SE3(ds * mcE[k-1])
                # print('g_SE\n',g_SE[k])

                # print("After")
                # visualize_frame(g_SE[k],mcV=mcV[k],mcE=mcE[k])
                # print('p :', g_SE[k][:, 3])
                Lambda_prime[k] = Lambda_prime_vf
                mcV_prime[k] = mcV_prime_vf


        elif self.integration_type == 'RK4':
            for k in range(1, self.N):
                s = s_points[k-1]
                ds = s_points[k] - s_points[k-1]
                F_ext = self.gravity_wrench(g_SE[k-1], mbM)
                Lambda_a = self.actuation_wrench(s, mcE[k-1])

                # ---- RK4 Integration ----

                # k1
                k1_Lambda, k1_mcV = self.semi_discretized_cosserat_equations(
                    s, mcE[k-1], mcV[k-1], mcE_h[k-1], mcV_h[k-1], mbM, mbK, mbD, F_ext, Lambda_a
                )

                # predict midpoints
                mcV_temp = mcV[k-1] + 0.5 * ds * k1_mcV
                Lambda_temp = Lambda[k-1] + 0.5 * ds * k1_Lambda
                mcE_temp = self.get_strain(s + 0.5*ds, Lambda_temp, Lambda_a, mbK, mbD, mcE_h[k-1])
                Lambda_a_temp = self.actuation_wrench(s + 0.5*ds, mcE_temp)
                F_ext_temp = self.gravity_wrench(g_SE[k-1], mbM)

                # k2
                k2_Lambda, k2_mcV = self.semi_discretized_cosserat_equations(
                    s + 0.5*ds, mcE_temp, mcV_temp, mcE_h[k-1], mcV_h[k-1], mbM, mbK, mbD, F_ext_temp, Lambda_a_temp
                )

                # k3
                mcV_temp = mcV[k-1] + 0.5 * ds * k2_mcV
                Lambda_temp = Lambda[k-1] + 0.5 * ds * k2_Lambda
                mcE_temp = self.get_strain(s + 0.5*ds, Lambda_temp, Lambda_a_temp, mbK, mbD, mcE_h[k-1])
                Lambda_a_temp = self.actuation_wrench(s + 0.5*ds, mcE_temp)
                k3_Lambda, k3_mcV = self.semi_discretized_cosserat_equations(
                    s + 0.5*ds, mcE_temp, mcV_temp, mcE_h[k-1], mcV_h[k-1], mbM, mbK, mbD, F_ext_temp, Lambda_a_temp
                )

                # k4
                mcV_temp = mcV[k-1] + ds * k3_mcV
                Lambda_temp = Lambda[k-1] + ds * k3_Lambda
                mcE_temp = self.get_strain(s + ds, Lambda_temp, Lambda_a_temp, mbK, mbD, mcE_h[k-1])
                Lambda_a_temp = self.actuation_wrench(s + ds, mcE_temp)
                F_ext_temp = self.gravity_wrench(g_SE[k-1], mbM)
                k4_Lambda, k4_mcV = self.semi_discretized_cosserat_equations(
                    s + ds, mcE_temp, mcV_temp, mcE_h[k-1], mcV_h[k-1], mbM, mbK, mbD, F_ext_temp, Lambda_a_temp
                )

                # update states
                Lambda[k] = Lambda[k-1] + (ds / 6.0) * (k1_Lambda + 2*k2_Lambda + 2*k3_Lambda + k4_Lambda)
                mcV[k] = mcV[k-1] + (ds / 6.0) * (k1_mcV + 2*k2_mcV + 2*k3_mcV + k4_mcV)

                # update strain
                mcE[k] = self.get_strain(s_points[k], Lambda[k], Lambda_a, mbK, mbD, mcE_h[k])

                # integrate configuration
                g_SE[k] = g_SE[k-1] @ exp_SE3(ds * mcE[k-1])

                # store derivatives for diagnostics
                Lambda_prime[k] = (k1_Lambda + 2*k2_Lambda + 2*k3_Lambda + k4_Lambda) / 6.0
                mcV_prime[k] = (k1_mcV + 2*k2_mcV + 2*k3_mcV + k4_mcV) / 6.0
        
        return g_SE, mcE, mcV, Lambda,  Lambda_prime, mcV_prime
        

    # ----------------------------------------------------------------------------------------------------------------------------- 
    # Implement Jacobians for the Newmark Beta:  C(nu0), A(nu0), B(nu0) 
    # -----------------------------------------------------------------------------------------------------------------------------
    def C_of_theta(self, theta0):
        """
        Build SE(3) pose g0 from theta0 = [Theta0(3), r0(3)].
        Returns 4x4 homogeneous matrix.
        """
        Theta0 = theta0[:3]
        r0 = theta0[3:]
        Rinc = exp_SO3(Theta0)                  # exp(Theta_hat)
        R0_new = self.R0_n @ Rinc               # R_{0,n} * exp(Theta_hat)
        g0 = np.eye(4)
        g0[:3, :3] = R0_new
        g0[:3, 3] = r0
        return g0

    def A_of_theta(self, theta0):
        """
        Base twist (velocity) from theta0.
        Appendix 3 notation -> implement using Newmark variables.
        Returns 6-vector eta0 = [Omega0; V0] in body frame consistent with paper.
        """
        Theta0 = theta0[:3]
        r0 = theta0[3:]

        # Newmark scalars
        dt = self.dt
        a = self.beta * dt * dt
        b = self.gamma * dt

        # previous-step predictor terms (computed from stored histories)
        kn = dt * self.Omega0_n + dt*dt*(0.5 - self.beta) * self.Omegadot0_n
        hn = self.Omega0_n + dt*(1 - self.gamma) * self.Omegadot0_n
        fn = self.r0_n + dt * self.rdot0_n + dt*dt*(0.5 - self.beta) * self.rddot0_n
        h_n = self.rdot0_n + dt*(1 - self.gamma) * self.rddot0_n

        # build the tilded coefficients from Appendix 3:
        # b_tilde = a^{-1}, a_tilde = b * b_tilde = b / a
        # f_tilde = hn - a_tilde * fn, h_tilde = -b_tilde * fn, k_tilde = ln - a_tilde * kn, l_tilde = -b_tilde * kn
        if abs(a) < 1e-16:
            raise RuntimeError("Newmark 'a' too small; check dt or beta.")
        b_tilde = 1.0 / a
        a_tilde = b * b_tilde
        f_tilde = h_n - a_tilde * fn
        h_tilde = -b_tilde * fn
        k_tilde = hn - a_tilde * kn
        l_tilde = -b_tilde * kn

        # rotation part Omega0
        Omega0 = a_tilde * Theta0 + k_tilde

        # translation part V0: careful with RT0^T piece in appendix:
        # They write RT0 = exp(Theta_hat)^T R_{0,n}^T, but simpler using R0_n and exp:
        Rinc = exp_SO3(Theta0)
        RT0 = Rinc.T @ self.R0_n.T   # transpose combination used in Appendix (RT0)
        # compute V0 in body frame as RT0 * (a_tilde * r0 + f_tilde)
        V0 = RT0 @ (a_tilde * r0 + f_tilde)

        eta0 = np.concatenate([Omega0, V0])
        return eta0

    def B_of_theta(self, theta0):
        """
        Base twist acceleration (Omega_dot, V_dot) from theta0.
        Implements Appendix 3 formula for B(theta0).
        """
        Theta0 = theta0[:3]
        r0 = theta0[3:]

        dt = self.dt
        a = self.beta * dt * dt
        b = self.gamma * dt

        kn = dt * self.Omega0_n + dt*dt*(0.5 - self.beta) * self.Omegadot0_n
        hn = self.Omega0_n + dt*(1 - self.gamma) * self.Omegadot0_n
        fn = self.r0_n + dt * self.rdot0_n + dt*dt*(0.5 - self.beta) * self.rddot0_n
        h_n = self.rdot0_n + dt*(1 - self.gamma) * self.rddot0_n

        if abs(a) < 1e-16:
            raise RuntimeError("Newmark 'a' too small; check dt or beta.")
        b_tilde = 1.0 / a
        a_tilde = b * b_tilde
        f_tilde = h_n - a_tilde * fn
        h_tilde = -b_tilde * fn
        k_tilde = hn - a_tilde * kn
        l_tilde = -b_tilde * kn

        # b_tilde and others used in Appendix 3:
        # Omega_dot = b_tilde * Theta0 + l_tilde
        Omega_dot = b_tilde * Theta0 + l_tilde

        # compute some intermediate terms for translation acceleration
        Rinc = exp_SO3(Theta0)
        RT0 = Rinc.T @ self.R0_n.T  # RT0 as in Appendix

        # compute V0 (using A_of_theta since needed for cross-product)
        mcV0 = self.A_of_theta(theta0)
        V0 = mcV0[3:]
        Omega0 = mcV0[:3]

        # Vdot = RT0( b_tilde*r0 + h_tilde ) + V0 x Omega0
        V_dot = RT0 @ (b_tilde * r0 + h_tilde) + np.cross(V0, Omega0)

        return np.concatenate([Omega_dot, V_dot])

   
    # =================================================================
    # Free-Free Solver
    # =================================================================
    def solve_bvp_free_free(self,
                            F_minus,       
                            F_plus,        
                            theta0_initial=None,
                            g_SE0=None,
                            mcE_prev=None, mcE_pprev=None, mcV_prev=None, mcV_pprev=None,
                            method='least_squares',
                            verbose=False):
        """
        Solve locomotor BVP by shooting on theta0 = [Theta0, r0].
        Includes scaling for numerical stability.
        """
        # Scaling Factors
        moment_scale = 1.0
        force_scale = 100.0 
        residual_scaler = np.array([1.0/moment_scale]*3 + [1.0/force_scale]*3)

        # default initial guess
        if theta0_initial is None:
            theta0_initial = np.zeros(6)
        # Left BC as initial condition
        Lambda0 = -F_minus.copy() 

        # objective function with scaling
        def objective_theta(theta0):
            try:
                # compute base pose and velocity from theta0
                g0 = self.C_of_theta(theta0)
                mcV0 = self.A_of_theta(theta0)
                
                # integrate along the rod
                g_SE, mcE, mcV, Lambda, Lambda_prime, mcV_prime = self.integrate_rod(
                    Lambda0, g0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev, self.s_points)
                
                # residual: Lambda(1) - F_plus
                raw_residual = Lambda[-1] - F_plus
                
                # Check for explosion
                if not np.all(np.isfinite(raw_residual)):
                     return np.ones(6) * 1e6
                
                # Return scaled residual
                return raw_residual * residual_scaler
                
            except Exception as e:
                return np.ones(6) * 1e6

        # Optimization
        try:
            res = least_squares(objective_theta, theta0_initial, method='lm', 
                              ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=100)
            theta_opt = res.x
        except Exception as e:
            print(f"Solver failed: {e}")
            theta_opt = theta0_initial

        # 4. Final Integration for solution fields
        g0_opt = self.C_of_theta(theta_opt)
        mcV0_opt = self.A_of_theta(theta_opt)
        g_SE, mcE, mcV, Lambda, Lambda_prime, mcV_prime = self.integrate_rod(
            Lambda0, g0_opt, mcV0_opt, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev, self.s_points)

        solution = {
            'g_SE': g_SE,
            'mcE': mcE,
            'mcV': mcV,
            'Lambda': Lambda,
            'Lambda_prime': Lambda_prime,
            'mcV_prime': mcV_prime,
            'theta_opt': theta_opt,
            'residual': Lambda[-1] - F_plus,
            's_points': self.s_points
        }
        return res, solution
    


    # ================================================================== Time Integration (static)

    # def time_integrate_cosserat_rod(self,g_SE0,mcV0,target_Lambda_tip,verbose=False):
    #     # integrate the rod over time 
    #     tspan = [self.t0]
    #     t0 = self.t0
    #     # interval for rod 
    #     s_points = np.linspace(0, self.L, self.N)
    #     # Initial conditions
    #     mcE_star = self.mcE_rest(0.0)  # Rest strain field
    #     # Initial conditions 
    #     # mcE0 = np.linalg.inv(mbK) @ Lambda0 + mcE_star  # Initial strain field
    #     # mcE_star = mcE0
    #     mcE_prev = np.array([mcE_star] * self.N)  # Previous strain field (for semi-discretization)
    #     mcE_pprev = np.array([mcE_star] * self.N)   # Previous strain field (for semi-discretization)
    #     mcV_prev = np.zeros((self.N,6))  # Previous strain field (for semi-discretization)
    #     mcV_pprev = np.zeros((self.N,6))  # Previous strain field (for semi-discretization)

    #     # g_SE_list = []  # List to store configurations at each time step
    #     # mcE_list = []  # List to store strain fields at each time step
    #     # Lambda_list = []  # List to store strain fields at each time step
    #     # Lambda0_list = []
    #     # mcV_list = []  # List to store velocity fields at each time step
    #     g_SE_list = np.zeros((self.N_steps,self.N,4,4))  # List to store configurations at each time step
    #     mcE_list = np.zeros((self.N_steps,self.N,6,))  # List to store strain fields at each time step
    #     Lambda_list = np.zeros((self.N_steps,self.N,6,))  # List to store strain fields at each time step
    #     Lambda0_list = np.zeros((self.N_steps,6,))
    #     mcV_list = np.zeros((self.N_steps,self.N,6,))  # List to store velocity fields at each time step
    #     # initial guess for stess field 
    #     mcE0=np.zeros((6,))
    #     # parameters
    #     mbM0, mbK0, mbD0 = self.soft_robot_parameters()

    #     Lambda0_initial_guess = np.zeros((6,))

    #     # Integrate the rod over time
    #     for i in range(self.N_steps):
    #         # solve the BVP
    #         result, solution = self.adaptive_shooting_method(
    #         g_SE0, mcV0, mcE_prev, mcE_pprev, mcV_prev, mcV_pprev,
    #         s_points, target_Lambda_tip,Lambda0_initial_guess=Lambda0_initial_guess,
    #         method='least_squares', verbose=True)

    #         g_SE3 = solution['g_SE']
    #         mcE = solution['mcE']
    #         mcV = solution['mcV']
    #         Lambda = solution['Lambda']
    #         optimal_Lambda0 = result.x
    #         # solution['Lambda_prime']
    #         # solution['mcV_prime']
    #         # reset the target boundary conditions
    #         target_Lambda_tip = np.zeros((6,))
    #         # Store the results
    #         g_SE_list[i,:,:,:] = g_SE3
    #         mcE_list[i,:,:] = mcE
    #         mcV_list[i,:,:] = mcV
    #         Lambda_list[i,:,:] = Lambda
    #         Lambda0_list[i,:] = optimal_Lambda0
            
    #         # Update initial conditions for the next time step (s = 0)
    #         mcE0 = mcE[0,:]
    #         mcV0 = mcV[0,:]
    #         g_SE0 = g_SE3[0,:,:]
    #         Lambda0_initial_guess = optimal_Lambda0
            
    #         # Store previous states for the next iteration
    #         mcE_pprev = mcE_prev.copy()
    #         mcE_prev = mcE.copy()
    #         mcV_pprev = mcV_prev.copy()
    #         mcV_prev = mcV.copy()
            
    #         t0 += self.dt
    #         print('t: ',t0)
    #         tspan.append(t0)
    #         # Visualize the configuration
    #         # visualize_rod(g_SE3, mcV=mcV,mcE=mcE)
    #         # self.plot_shooting_results(solution, target_Lambda_tip)

    #         # check if the rod has moved 
    #         # print('g_SE@ g_SE_inv:', g_SE[-1,:,:]@SE3_inv(g_SE_list[0][-1,:,:]))
    # #         plot_shooting_results(solution, target_Lambda_tip)
        
    #     return g_SE_list, mcE_list, mcV_list, Lambda_list, Lambda0_list
    
    # test initial force 
    def force_tent(self,t,tf=1.5):
        c0 = 15.7
        if t <= tf/2:
            return c0*(2/tf)*t
        elif t > tf/2 and t<=tf:
            return c0*2/tf*(tf-t)
        else:
            return 0.0
        
    def test_applied_force(self,t,tf=1.5):
        # time varying force 
        F_ext = np.zeros((6,))
        F_ext[0] = 0.0
        F_ext[1] = self.force_tent(t,tf=tf)
        F_ext[2] = self.force_tent(t,tf=tf)/40
        F_ext[5] = self.force_tent(t,tf=tf)
        return F_ext
    
    def get_jacobians_free_free(self, theta0):
        # Unpack theta0
        Theta0 = theta0[:3]
        r0 = theta0[3:]
        
        # Recompute kinematics to get current state
        g0 = self.C_of_theta(theta0)
        mcV0 = self.A_of_theta(theta0)
        mcVdot0 = self.B_of_theta(theta0)
        
        Omega0 = mcV0[:3]
        V0 = mcV0[3:]
        Omegadot0 = mcVdot0[:3]
        Vdot0 = mcVdot0[3:]
        
        # Newmark constants (tilde vars from paper)
        dt = self.dt
        a = self.beta * dt**2
        b = self.gamma * dt
        b_tilde = 1.0/a
        a_tilde = b/a
        
        # 1. dC/dtheta0  (eq 112)
        RT0 = exp_SO3(Theta0).T @ self.R0_n.T # From eq 111 text
        T_mat = T_SO3(Theta0) 
        
        dC_dtheta = np.zeros((6,6))
        dC_dtheta[:3, :3] = T_mat
        dC_dtheta[3:, 3:] = RT0

        # 2. dA/dtheta0  (eq 113 top)
        dA_dtheta = np.zeros((6,6))
        dA_dtheta[:3, :3] = a_tilde * np.eye(3)
        dA_dtheta[3:, :3] = hat_SO3(V0) @ T_mat  # Cross term
        dA_dtheta[3:, 3:] = a_tilde * RT0

        # 3. dB/dtheta0  (eq 113 bottom)
        dB_dtheta = np.zeros((6,6))
        
        # d(Omega_dot)/dTheta = b_tilde * I
        dB_dtheta[:3, :3] = b_tilde * np.eye(3)
        
        # d(V_dot)/dr0
        # V_dot term involves (b_tilde * I - a_tilde * Omega_hat) * RT0
        dB_dtheta[3:, 3:] = (b_tilde * np.eye(3) - a_tilde * hat_SO3(Omega0)) @ RT0
        
        # d(V_dot)/dTheta
        A0 = Vdot0 + np.cross(Omega0, V0) # Spatial acceleration
        term1 = hat_SO3(A0) - hat_SO3(Omega0) @ hat_SO3(V0)
        dB_dtheta[3:, :3] = term1 @ T_mat + a_tilde * hat_SO3(V0)

        return dC_dtheta, dA_dtheta, dB_dtheta


    # =================================================================  integrate free free
    def time_integrate_free_free(self, F_minus, F_plus, verbose=False):
        tspan = [self.t0]
        t0 = self.t0
        mcE_starr = self.mcE_star_vec()
        
        # Initial History (Cold Start)
        mcE_star = self.mcE_rest(0.0)
        mcE_prev = np.array([mcE_star] * self.N)
        mcE_pprev = np.array([mcE_star] * self.N)
        mcV_prev = np.zeros((self.N,6))
        mcV_pprev = np.zeros((self.N,6))

        # Storage
        g_SE_list = np.zeros((self.N_steps, self.N, 4, 4))
        mcE_list = np.zeros((self.N_steps, self.N, 6))
        mcV_list = np.zeros((self.N_steps, self.N, 6))
        Lambda_list = np.zeros((self.N_steps, self.N, 6))
        theta0_list = np.zeros((self.N_steps, 6))

        # Initial Guess for Base Motion
        theta0_guess = np.zeros(6) 

        for i in range(self.N_steps):
            if verbose: print(f"Step {i}, t={t0:.4f}")

            # Solve BVP
            # applied force
            if self.toss_flag:
                F_plus = self.test_applied_force(t0)
            

            result, solution = self.solve_bvp_free_free(
                F_minus=F_minus, # Use the argument passed to the function
                F_plus=F_plus,
                theta0_initial=theta0_guess,
                mcE_prev=mcE_prev, mcE_pprev=mcE_pprev, 
                mcV_prev=mcV_prev, mcV_pprev=mcV_pprev,
                verbose=verbose
            )

            # unpack optimal solution (next time step)
            theta_opt = solution['theta_opt']
            g_SE3 = solution['g_SE']
            mcE = solution['mcE']
            mcV = solution['mcV']
            Lambda = solution['Lambda']
            
            # compute the kinematics of the base
            g0_opt = self.C_of_theta(theta_opt)
            mcV0_opt = self.A_of_theta(theta_opt)
            mcVdot0_opt = self.B_of_theta(theta_opt)

            # update newmark history
            self.R0_n = g0_opt[:3, :3]
            self.r0_n = g0_opt[:3, 3]
            self.Omega0_n = mcV0_opt[:3]
            self.rdot0_n = self.R0_n @ mcV0_opt[3:]
            self.Omegadot0_n = mcVdot0_opt[:3]
            
            # coriolis term to world-frame acceleration
            V_body = mcV0_opt[3:]
            Omega_body = mcV0_opt[:3]
            V_dot_body = mcVdot0_opt[3:]
            # acceleration in world frame = R * (V_dot + Omega x V)
            self.rddot0_n = self.R0_n @ (V_dot_body + np.cross(Omega_body, V_body))

            # store variables
            g_SE_list[i] = g_SE3
            mcE_list[i] = mcE - mcE_starr
            mcV_list[i] = mcV
            Lambda_list[i] = Lambda
            theta0_list[i] = theta_opt

            mcE_pprev = mcE_prev.copy()
            mcE_prev = mcE.copy()
            mcV_pprev = mcV_prev.copy()
            mcV_prev = mcV.copy()
            
            # next step
            theta0_guess = theta_opt 
            t0 += self.dt
            tspan.append(t0)

        self.tspan = tspan
        return g_SE_list, mcE_list, mcV_list, Lambda_list, theta0_list
    
    # ================================================================== Plotting Visualization
    def plot_shooting_results(self,solution, target_Lambda_tip=None):
        """
        Plot the results of the shooting method solution.
        
        Args:
            solution: Solution dictionary from solve_bvp_shooting_method
            target_Lambda_tip: Target tip stress for comparison
        """
        
        s_points = solution['s_points']
        Lambda = solution['Lambda']
        mcE = solution['mcE']
        mcV = solution['mcV']
        g_SE = solution['g_SE']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot stress components
        axes[0, 0].plot(s_points, Lambda[:, :3], label=['τ₁', 'τ₂', 'τ₃'])
        axes[0, 0].set_title('Moment Components')
        axes[0, 0].set_xlabel('Arc length s')
        axes[0, 0].set_ylabel('Moment')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(s_points, Lambda[:, 3:], label=['F₁', 'F₂', 'F₃'])
        axes[0, 1].set_title('Force Components')
        axes[0, 1].set_xlabel('Arc length s')
        axes[0, 1].set_ylabel('Force')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot strain components
        axes[0, 2].plot(s_points, mcE[:, :3], label=['κ₁', 'κ₂', 'κ₃'])
        axes[0, 2].set_title('Curvature Components')
        axes[0, 2].set_xlabel('Arc length s')
        axes[0, 2].set_ylabel('Curvature')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        axes[1, 0].plot(s_points, mcE[:, 3:], label=['γ₁', 'γ₂', 'γ₃'])
        axes[1, 0].set_title('Shear/Extension Components')
        axes[1, 0].set_xlabel('Arc length s')
        axes[1, 0].set_ylabel('Strain')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot rod configuration (centerline)
        positions = g_SE[:, :3, 3]
        axes[1, 1].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        axes[1, 1].set_title('Rod Centerline (XY view)')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].amcEs('equal')
        axes[1, 1].grid(True)
        
        # Plot 3D configuration
        ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax_3d.set_title('Rod Centerline (3D)')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        
        # Add target tip stress comparison if provided
        if target_Lambda_tip is not None:
            final_Lambda = Lambda[-1]
            print("\nBoundary condition comparison:")
            print(f"Target tip stress: {self.target_Lambda_tip}")
            print(f"Achieved tip stress: {final_Lambda}")
            print(f"Error: {final_Lambda - self.target_Lambda_tip}")
            print(f"Error norm: {np.linalg.norm(final_Lambda - self.target_Lambda_tip)}")
        
        plt.tight_layout()
        plt.show()

    def animate_cosserat_rod(self,g_SE_list, mcE_list, mcV_list, Lambda_list, 
                        rigid_motion_amcEs=None, set_axes_equal=None):
        """
        Animate the Cosserat rod simulation results
        
        Parameters:
        - g_SE_list: (N_steps, N, 4, 4) - SE(3) transformations over time
        - mcE_list: (N_steps, N, 6) - strain fields over time  
        - mcV_list: (N_steps, N, 6) - velocity fields over time
        - Lambda_list: (N_steps, N, 6) - force/moment fields over time
        - dt: time step
        - rigid_motion_amcEs: function to extract axes from SE(3) matrix
        - set_axes_equal: function to set equal amcEs scaling
        """
        
        N_steps, N, _, _ = g_SE_list.shape
        
        # Create figure and amcEs
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Pre-compute rod backbone positions for all time steps
        all_positions = np.zeros((N_steps, N, 3))
        
        for t in range(N_steps):
            for i in range(N):
                # Extract position from SE(3) matrix
                all_positions[t, i, :] = g_SE_list[t, i, :3, 3]
        
        # Set up plot limits based on all data
        all_pos_flat = all_positions.reshape(-1, 3)
        margin = 0.2
        x_range = [all_pos_flat[:, 0].min() - margin, all_pos_flat[:, 0].max() + margin]
        y_range = [all_pos_flat[:, 1].min() - margin, all_pos_flat[:, 1].max() + margin]
        z_range = [all_pos_flat[:, 2].min() - margin, all_pos_flat[:, 2].max() + margin]
        
        def animate(frame):
            ax.clear()
            
            # Current time step
            t = frame
            current_time = t * self.dt
            
            # Extract current rod configuration
            positions = all_positions[t]
            mcE = mcE_list[t]
            mcV = mcV_list[t]
            Lambda = Lambda_list[t]
            
            # Plot rod backbone
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'ko-', linewidth=3, markersize=6, label='Rod backbone')
            
            # Plot coordinate frames along the rod (every few points to avoid clutter)
            step = max(1, N // 20)  # Show ~8 coordinate frames
            for i in range(0, N, step):
                pos = positions[i]
                
                if rigid_motion_axis is not None:
                    try:
                        x_axis, y_axis, z_axis, _ = rigid_motion_axis(g_SE_list[t, i])
                        scale = 0.1
                        ax.quiver(*pos, *(x_axis * scale), color='red', alpha=0.7, arrow_length_ratio=0.15)
                        ax.quiver(*pos, *(y_axis * scale), color='green', alpha=0.7, arrow_length_ratio=0.15)
                        ax.quiver(*pos, *(z_axis * scale), color='blue', alpha=0.7, arrow_length_ratio=0.15)
                    except:
                        # Fallback: extract rotation matrix directly
                        R = g_SE_list[t, i, :3, :3]
                        scale = 0.1
                        ax.quiver(*pos, *(R[:, 0] * scale), color='red', alpha=0.7, arrow_length_ratio=0.15)
                        ax.quiver(*pos, *(R[:, 1] * scale), color='green', alpha=0.7, arrow_length_ratio=0.15)
                        ax.quiver(*pos, *(R[:, 2] * scale), color='blue', alpha=0.7, arrow_length_ratio=0.15)
            
            # Plot velocity field (every few points)
            for i in range(0, N, step):
                pos = positions[i]
                velocity = mcV[i, 3:6]  # Linear velocity
                angular_vel = mcV[i, :3]  # Angular velocity
                
                if np.linalg.norm(velocity) > 1e-6:
                    ax.quiver(*pos, *velocity, color='purple', alpha=0.8, 
                            arrow_length_ratio=0.1, linewidth=2)
                if np.linalg.norm(angular_vel) > 1e-6:
                    ax.quiver(*pos, *angular_vel, color='orange', alpha=0.8, 
                            arrow_length_ratio=0.1, linewidth=2)
            
            # Plot strain field (every few points)
            for i in range(0, N, step):
                pos = positions[i]
                strain_linear = mcE[i, 3:6]  # Linear strain
                strain_angular = mcE[i, :3]  # Angular strain
                
                if np.linalg.norm(strain_linear) > 1e-6:
                    ax.quiver(*pos, *strain_linear, color='cyan', alpha=0.8, 
                            arrow_length_ratio=0.15, linewidth=2)
                if np.linalg.norm(strain_angular) > 1e-6:
                    ax.quiver(*pos, *strain_angular, color='pink', alpha=0.8, 
                            arrow_length_ratio=0.15, linewidth=2)
            
            # Highlight tip of the rod
            tip_pos = positions[-1]
            ax.scatter(*tip_pos, color='red', s=100, alpha=0.9, label='Rod tip')
            
            # Show trajectory trail of the tip
            if t > 0:
                tip_trail = all_positions[:t+1, -1, :]
                ax.plot(tip_trail[:, 0], tip_trail[:, 1], tip_trail[:, 2], 
                    'r--', alpha=0.5, linewidth=2, label='Tip trajectory')
            
            # Set plot properties
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if set_axes_equal is not None:
                set_axes_equal(ax)
            
            ax.set_title(f'Cosserat Rod Dynamics\nTime: {current_time:.3f}s (Step {t+1}/{N_steps})')
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=N_steps, 
                                    interval=max(50, int(self.dt*1000)), blit=False, repeat=True)
        
        return fig, anim

    def analyze_rod_dynamics(self,g_SE_list, mcE_list, mcV_list):
        """
        Create analysis plots of the rod dynamics
        """
        N_steps, N = g_SE_list.shape[:2]
        time_array = np.arange(N_steps) * self.dt
        
        # Extract key metrics
        tip_positions = np.zeros((N_steps, 3))
        tip_velocities = np.zeros((N_steps, 3))
        rod_length = np.zeros(N_steps)
        
        for t in range(N_steps):
            tip_positions[t] = g_SE_list[t, -1, :3, 3]
            tip_velocities[t] = mcV_list[t, -1, 3:6]
            
            # Calculate rod length (arc length)
            positions = g_SE_list[t, :, :3, 3]
            differences = np.diff(positions, axis=0)
            segment_lengths = np.linalg.norm(differences, axis=1)
            rod_length[t] = np.sum(segment_lengths)
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Tip position
        axes[0,0].plot(time_array, tip_positions[:, 0], 'r-', label='X')
        axes[0,0].plot(time_array, tip_positions[:, 1], 'g-', label='Y')  
        axes[0,0].plot(time_array, tip_positions[:, 2], 'b-', label='Z')
        axes[0,0].set_title('Tip Position vs Time')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Position')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Tip velocity
        axes[0,1].plot(time_array, tip_velocities[:, 0], 'r-', label='Vx')
        axes[0,1].plot(time_array, tip_velocities[:, 1], 'g-', label='Vy')
        axes[0,1].plot(time_array, tip_velocities[:, 2], 'b-', label='Vz')
        axes[0,1].set_title('Tip Velocity vs Time')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Velocity')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Rod length
        axes[1,0].plot(time_array, rod_length, 'k-', linewidth=2)
        axes[1,0].set_title('Rod Length vs Time')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Length')
        axes[1,0].grid(True, alpha=0.3)
        
        # Tip trajectory in 3D
        axes[1,1].remove()
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        ax_3d.plot(tip_positions[:, 0], tip_positions[:, 1], tip_positions[:, 2], 
                'b-', linewidth=2, alpha=0.7)
        ax_3d.scatter(tip_positions[0, 0], tip_positions[0, 1], tip_positions[0, 2], 
                    color='green', s=100, label='Start')
        ax_3d.scatter(tip_positions[-1, 0], tip_positions[-1, 1], tip_positions[-1, 2], 
                    color='red', s=100, label='End')
        ax_3d.set_title('3D Tip Trajectory')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.legend()
        
        plt.tight_layout()
        return fig

    def plot_shooting_results(self,solution, target_Lambda_tip=None):
        """
        Plot the results of the shooting method solution.
        
        Args:
            solution: Solution dictionary from solve_bvp_shooting_method
            target_Lambda_tip: Target tip stress for comparison
        """
        
        s_points = solution['s_points']
        Lambda = solution['Lambda']
        mcE = solution['mcE']
        mcV = solution['mcV']
        g_SE = solution['g_SE']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot stress components
        axes[0, 0].plot(s_points, Lambda[:, :3], label=['τ₁', 'τ₂', 'τ₃'])
        axes[0, 0].set_title('Moment Components')
        axes[0, 0].set_xlabel('Arc length s')
        axes[0, 0].set_ylabel('Moment')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(s_points, Lambda[:, 3:], label=['F₁', 'F₂', 'F₃'])
        axes[0, 1].set_title('Force Components')
        axes[0, 1].set_xlabel('Arc length s')
        axes[0, 1].set_ylabel('Force')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot strain components
        axes[0, 2].plot(s_points, mcE[:, :3], label=['κ₁', 'κ₂', 'κ₃'])
        axes[0, 2].set_title('Curvature Components')
        axes[0, 2].set_xlabel('Arc length s')
        axes[0, 2].set_ylabel('Curvature')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        axes[1, 0].plot(s_points, mcE[:, 3:], label=['γ₁', 'γ₂', 'γ₃'])
        axes[1, 0].set_title('Shear/Extension Components')
        axes[1, 0].set_xlabel('Arc length s')
        axes[1, 0].set_ylabel('Strain')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot rod configuration (centerline)
        positions = g_SE[:, :3, 3]
        axes[1, 1].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        axes[1, 1].set_title('Rod Centerline (XY view)')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].axis('equal')
        axes[1, 1].grid(True)
        
        # Plot 3D configuration
        ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax_3d.set_title('Rod Centerline (3D)')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        
        # Add target tip stress comparison if provided
        if target_Lambda_tip is not None:
            final_Lambda = Lambda[-1]
            print("\nBoundary condition comparison:")
            print(f"Target tip stress: {target_Lambda_tip}")
            print(f"Achieved tip stress: {final_Lambda}")
            print(f"Error: {final_Lambda - target_Lambda_tip}")
            print(f"Error norm: {np.linalg.norm(final_Lambda - target_Lambda_tip)}")
        
        plt.tight_layout()
        plt.show()