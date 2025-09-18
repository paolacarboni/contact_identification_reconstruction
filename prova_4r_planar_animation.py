import sys
sys.path.insert(0, "/home/pcarboni/MARR/TESI/simulazioni_py/robotics-toolbox-python")

import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH, RevoluteMDH, jtraj
import roboticstoolbox.frne as frne
from my_DHRobot import my_DHRobot
from my_DHRobot_fext import my_DHRobot_fext
from residuals import residuals
from math import pi
from solve_planar import SolvePlanarSystem
import sympy as sp


plt.style.use("ggplot")  # ✅ same style as your Anthro3R

class Planar_4r:
    def __init__(self, link_lengths, link_masses, link_radiuses):
        self.l = link_lengths
        self.m = link_masses
        self.r = link_radiuses
        self.robot = self._create_robot()

    def _create_robot(self):
        L = []
        for i in range(4):
            # link center of mass
            r_i = [-self.l[i] / 2, 0, 0]

            # crude inertia
            Ixx = 1/2 * self.r[i]**2
            Iyy = 1/12 * (3*self.r[i]**2 + self.l[i]**2)
            Izz = Iyy
            I_i = self.m[i] * np.diag([Ixx, Iyy, Izz])

            link = RevoluteDH(
                a=self.l[i], alpha=0, d=0,
                m=self.m[i], r=r_i, I=I_i, Jm=0, G=1
            )
            L.append(link)

        return my_DHRobot_fext(L, name='Planar_4r')

    def plot(self, q=None, block=True, style="ggplot"):
        if q is None:
            q = [0, 0, 0, 0]

        plt.style.use(style)

        # forward kinematics for each link
        T = np.eye(4)
        points = [[0, 0]]
        for i, theta in enumerate(q):
            T = T @ self.robot.links[i].A(theta).A
            points.append([T[0, 3], T[1, 3]])

        points = np.array(points)

        plt.figure(figsize=(6,6))
        plt.plot(points[:,0], points[:,1], '-o', linewidth=3, markersize=8)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Planar 4R Robot")
        plt.axis("equal")
        plt.grid(True)
        plt.show(block=block)


def init_custom_3d():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 2])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Anthropomorphic 3R Arm")
    return fig, ax

def draw_frame(ax, T, length=0.1):
    origin = T.t
    R = T.R
    ax.quiver(*origin, *R[:, 0] * length, color='r', linewidth=1.2)
    ax.quiver(*origin, *R[:, 1] * length, color='g', linewidth=1.2)
    ax.quiver(*origin, *R[:, 2] * length, color='b', linewidth=1.2)

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale so that arrows are not distorted."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a cube, so all axes are scaled equally
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def update_custom_3d(ax, robot, q):
    ax.cla()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 0.2])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Anthropomorphic 3R Arm")

    Ts = robot.fkine_all(q)
    xs, ys, zs = [], [], []
    for T in Ts:
        p = T.t
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])
        draw_frame(ax, T, length=0.3)
    ax.plot(xs, ys, zs, '-o', color='royalblue', markersize=4, linewidth=2)
    ax.scatter(xs[-1], ys[-1], zs[-1], color='red', s=80, edgecolors='k')
    set_axes_equal(ax)
    plt.pause(0.001)

link_lengths = [0.5, 0.5, 0.5, 0.5]
link_masses  = [5, 5, 5, 5]
link_radiuses = [0.2, 0.2, 0.2, 0.2]

p4r = Planar_4r(link_lengths, link_masses, link_radiuses)
robot = p4r.robot

# Setup figure
fig, ax = init_custom_3d()

# # Animate a trajectory
# q_traj = np.linspace(0, np.pi/2, 200)   # simple joint motion
# for q1 in q_traj:
#     q = [q1, -q1/2, q1/3, -q1/4]
#     update_custom_3d(ax, robot, q)


def draw_frame(ax, T, length=0.05):
    origin = T.t
    R = T.R
    ax.quiver(*origin, *R[:, 0] * length, color="r", linewidth=1.2)
    ax.quiver(*origin, *R[:, 1] * length, color="g", linewidth=1.2)
    ax.quiver(*origin, *R[:, 2] * length, color="b", linewidth=1.2)


def update_custom_3d(ax, robot, q, reach, margin):
    ax.cla()
    ax.set_xlim([-reach - margin, reach + margin])
    ax.set_ylim([-reach - margin, reach + margin])
    ax.set_zlim([-0.0, 0.5])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Planar 4R Arm")

    Ts = robot.fkine_all(q)
    xs, ys, zs = [], [], []
    for T in Ts:
        p = T.t
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])
        draw_frame(ax, T)
    ax.plot(xs, ys, zs, "-o", color="royalblue", markersize=6, linewidth=2)
    ax.scatter(xs[-1], ys[-1], zs[-1], color="red", s=80, edgecolors="k")
    plt.pause(0.001)


# ---------------- Run animation -------------- #
# if __name__ == "__main__":
#     link_lengths = [0.5, 0.5, 0.5, 0.5]
#     link_masses = [5, 5, 5, 5]
#     link_radiuses = [0.05, 0.05, 0.05, 0.05]

#     p4r = Planar_4r(link_lengths, link_masses, link_radiuses)
#     robot = p4r.robot

#     # Setup figure
#     fig, ax, reach, margin = init_custom_3d(robot)

#     # Animate a simple trajectory
#     q_traj = np.linspace(0, np.pi / 2, 200)
#     for q1 in q_traj:
#         q = [q1, -q1 / 2, q1 / 3, -q1 / 4]
#         update_custom_3d(ax, robot, q, reach, margin)

n = robot.n # number of joints
# Suppose n joints
fext_links = [[0,0,0, 0,0,0] for _ in range(n)] # Fx Fy Fz Mx My Mz
pext_links = [[0,0,0] for _ in range(n)] # px py pz

# Apply a 10 N force along +x at 0.2 m along link 3 (index 2), no moment
#fext_links[2] = [0,0,0, 0,0,0]
#pext_links[2] = [-0.1,0,0]
# --------------------------
# Simulation config
# --------------------------
#DT = 0.002
DT = 0.002
T  = .3
Kp = 30 * np.diag([1, 1, 1,1 ])
Kd = 30 * np.diag([1, 1, 1, 1])
K_0 = 20 * np.diag([1, 1, 1, 1]) # residual gain

q0 = np.array([0, 0, 0, 0])
qf = np.array([0, 0, 0, 0])

time = np.arange(0, T, DT)
N = len(time)

#gravity
#g_0 = np.array([0, 0, -9.81])
g_0 = np.array([0, 0, 0])
# Trajectory
traj = jtraj(q0, qf, N)
q_d, qd_d, qdd_d = traj.q, traj.qd, traj.qdd

# Logs
q = q0.copy()
qd = np.zeros(n)

q_log   = np.zeros((N, n))
qd_log  = np.zeros((N, n))
tau_log = np.zeros((N, n))
tau_prime_log = np.zeros((N, n))
tau_ext_log = np.zeros((N, n))
res_log = np.zeros((N, n))

# reconstructed values, single contact force
F_x_log = np.zeros(N)
F_y_log = np.zeros(N)
l_log = np.zeros(N)

#F_ext_ee = np.array([0,0,-30, 0, 0, 0])

fext_links = np.zeros((n, 3), dtype=np.float64)
fext_links_zeros = np.zeros((n, 3), dtype=np.float64)
mext_links = np.zeros((n, 3), dtype=np.float64)
pext_links = np.zeros((n, 3), dtype=np.float64)
pext_links_zeros = np.zeros((n, 3), dtype=np.float64)
res = np.zeros((n), dtype=np.float64)
p_hat = np.zeros((n), dtype=np.float64)

tau_ext_try_2 = np.zeros((n), dtype=np.float64)
tau_ext_try_3 = np.zeros((n), dtype=np.float64)

# definizione variabili forze esterne e punti di contatto
F1_ext = np.array([0, 0, 0])
P1_ext = np.array([0, 0, 0])

F2_ext = np.array([0, 0, 0])
P2_ext = np.array([0, 0, 0])

F3_ext = np.array([0, 0, 0])
P3_ext = np.array([0, 0, 0])

F4_ext = np.array([0, 0, 0])
P4_ext = np.array([0, 0, 0])

# valori nominali di forze esterne e punti di contatto 
F1_ext_ = np.array([0,-100, 0])
P1_ext_ = np.array([-0.1,0,0])

F2_ext_ = np.array([0,-100, 0])
P2_ext_ = np.array([-0.1,0,0])

F3_ext_ = np.array([0,-100,0])
P3_ext_ = np.array([-0.1,0,0])

F4_ext_ = np.array([0,-100,0])
P4_ext_ = np.array([-0.1,0,0])   

#time_interval_1 = np.array([0, 0.5])
time_interval_1 = np.array([0.1, 0.98])
time_interval_2 = np.array([0.2, 0.3])
time_interval_3 = np.array([0.2, .3])
time_interval_4 = np.array([1.2, 1.5])

#case for single force applied
num_forces = 1 #[1, 2] how many external forces are applied
case_single = 3 # [1, 2, 3, 4] on which link is the force applied
case_double = 34 # [14, 24, 34, 44] on which link is the force applied

if num_forces == 1: 
    case = case_single
else:
    case = case_double

# SINGLE FORCE CASE
if num_forces ==1 and case_single == 1:
    bool_f1 = True
    bool_f2 = False
    bool_f3 = False
    bool_f4 = False
    F1_ext = F1_ext_
    P1_ext = P1_ext_
elif num_forces ==1 and case_single == 2:
    bool_f1 = False
    bool_f2 = True
    bool_f3 = False
    bool_f4 = False
    F2_ext = F2_ext_
    P2_ext = P2_ext_
elif num_forces ==1 and case_single == 3:
    bool_f1 = False
    bool_f2 = False
    bool_f3 = True
    bool_f4 = False
    F3_ext = F3_ext_
    P3_ext = P3_ext_
elif num_forces ==1 and case_single == 4:
    bool_f1 = False
    bool_f2 = False
    bool_f3 = False
    bool_f4 = True
    F4_ext = F4_ext_
    P4_ext = P4_ext_    

# DOUBLE FORCE CASE
if num_forces ==2 and case_double == 14:
    bool_f1 = True
    bool_f2 = False
    bool_f3 = False
    bool_f4 = True
    F1_ext = F1_ext_
    P1_ext = P1_ext_
    F4_ext = F4_ext_
    P4_ext = P4_ext_
elif num_forces ==2 and case_double == 24:
    bool_f1 = False
    bool_f2 = True
    bool_f3 = False
    bool_f4 = True
    F2_ext = F2_ext_
    P2_ext = P2_ext_
    F4_ext = F4_ext_
    P4_ext = P4_ext_
elif num_forces ==2 and case_double == 34:
    bool_f1 = False
    bool_f2 = False
    bool_f3 = True
    bool_f4 = True
    F3_ext = F3_ext_
    P3_ext = P3_ext_
    F4_ext = F4_ext_
    P4_ext = P4_ext_
elif num_forces ==2 and case_double == 44:
    bool_f1 = False
    bool_f2 = False
    bool_f3 = False
    bool_f4 = True
    F4_ext = F4_ext_
    P4_ext = P4_ext_


def rk4_step(q, qd, tau, tau_ext, tau_prime, M, DT):
    """
    One RK4 step for rigid-body dynamics
    q, qd : current position/velocity (shape (n,))
    tau, tau_prime : torques (shape (n,))
    M : inertia matrix (n x n)
    DT : timestep
    """
    n = len(q)

    def f(y):
        q = y[:n]
        qd = y[n:]
        qdd = np.linalg.pinv(M) @ (tau + tau_ext - tau_prime)
        #qdd = np.linalg.pinv(M) @ (tau- tau_prime)
        return np.concatenate([qd, qdd])

    y = np.concatenate([q, qd])

    k1 = f(y)
    k2 = f(y + 0.5 * DT * k1)
    k3 = f(y + 0.5 * DT * k2)
    k4 = f(y + DT * k3)

    y_new = y + DT/6 * (k1 + 2*k2 + 2*k3 + k4)

    return y_new[:n], y_new[n:]

q_test = np.array([0.0, 0.0, 0.0])
link_i = 2                  # apply on link 3 (0-based)
f_local = np.array([0, 0, -300])   # downward in link frame
p_local = np.array([-0.1, 0, 0])   # offset from link origin

solver = SolvePlanarSystem(num_forces, case)

# tau_jac = compute_tau_from_force(robot, q_test, link_i, f_local, p_local)
# print("Torque from Jacobian method:", tau_jac)

# --------------------------
# Simulation loop
# --------------------------
ANIMATE = False#
if ANIMATE:
    fig, ax = init_custom_3d()

for k, t in enumerate(time):
    print('t', t)
    #print('entered enumerate time')
    q_ref, qd_ref, qdd_ref = q_d[k], qd_d[k], qdd_d[k]
    e, edot = q_ref - q, qd_ref - qd

    # Newton–Euler feedforward + PD feedback
    #tau_ff = robot.rne(q_ref, qd_ref, qdd_ref, gravity = g_0, ext_forces=Fe, ext_moments=Ne, ext_points=Re)
    # print(' tau ff: ', tau_ff)
    
    tau_fb = Kp @ e + Kd @ edot
    #tau = tau_ff + tau_fb
    tau = tau_fb

    # external forces
    if bool_f1 and t > time_interval_1[0] and t < time_interval_1[1]:
        fext_links[0] = F1_ext
        pext_links[0] = P1_ext
    else:
        fext_links[0] = np.zeros(3)
        pext_links[0] = np.zeros(3)
        #bool_f1 = False
    if bool_f2 and t > time_interval_2[0] and t < time_interval_2[1]:
        fext_links[1] = F2_ext
        pext_links[1] = P2_ext
    else:
        fext_links[1] = np.zeros(3)
        pext_links[1] = np.zeros(3)
        #bool_f2 = False
    if bool_f3 and t > time_interval_3[0] and t < time_interval_3[1]:
        fext_links[2] = F3_ext
        pext_links[2] = P3_ext
    else:
        fext_links[2] = np.zeros(3)
        pext_links[2] = np.zeros(3)
    if bool_f4 and t > time_interval_4[0] and t < time_interval_4[1]:
        fext_links[3] = F4_ext
        pext_links[3] = P4_ext
    else:
        fext_links[3] = np.zeros(3)
        pext_links[3] = np.zeros(3)

    fext_array = np.array(fext_links)
    totals = fext_array.sum(axis=0)
    Fx_tot = totals[0]
    Fy_tot = totals[1]
    # Forward dynamics
    #M = robot.inertia(q)
    #C = robot.coriolis(q, qd)
    #g = robot.gravload(q)

    # stima matrice di inerzia e la sua derivata
    M_temp = np.zeros((n, n))
    M_temp_next = np.zeros((n,n))
    q_next = q + qd*DT
    for i in range(n):
        #print('entered M estimation')
        qdd = np.zeros(n)
        qdd[i] = 1
        # print('qdd: ', qdd)
        #print('q: ', q)
        #print('qdd: ', qdd)
        #print('GRAVITY SHOULD BE ZERO')
        m_i = robot.rne(q, np.zeros(n), qdd, gravity=np.zeros(3))
        #print('m_i: ', m_i)
        M_temp[:, i] = m_i
        #print("GRAVITY SHOULD BE ZERO")
        m_next_i = robot.rne(q_next, np.zeros(n), qdd, gravity=np.zeros(3))
        M_temp_next[:,i] = m_next_i
    M = 0.5 * (M_temp + M_temp.T)
    M_next = 0.5 * (M_temp_next + M_temp_next.T)

    M_dot = (M_next - M)/DT

    #print('M estimated')
    #stima di tau_prime
    #print("GRAVITY SHOULD NOT BE ZERO")
    tau_prime = robot.rne(q, qd, np.zeros(n), gravity = g_0, fext_links = fext_links_zeros, pext_links = pext_links)#, ext_forces=Fe, ext_moments=Ne, ext_points=Re)
    #tau_prime_ext = robot.rne(q, qd, np.zeros(n), gravity = g_0, fext_links = fext_links_zeros, pext_links = pext_links_zeros) # correct according to theory 
    #rhs = tau - C @ qd - g

    #print('tau tot, fext != 0')
    tau_tot = robot.rne(q, qd, qdd, gravity = g_0, fext_links = fext_links, pext_links = pext_links)
    tau_no_forces = robot.rne(q, qd, qdd, gravity = g_0, fext_links = fext_links_zeros, pext_links = pext_links_zeros) #fext_links = fext_links_zeros
    tau_ext = -(tau_tot - tau_no_forces)
    #print('tau_ext', tau_ext)


    # if bool_f2 and t > time_interval_2[0] and t < time_interval_2[1]:
    #     tau_ext_try_2 = compute_tau_from_force(robot, q, 1, fext_links[1], pext_links[1])
    # if bool_f3 and t > time_interval_3[0] and t < time_interval_3[1]:
    #     tau_ext_try_3 = compute_tau_from_force(robot, q, 2, fext_links[2], pext_links[2])

    # tau_ext_try = tau_ext_try_2 + tau_ext_try_3
    # print('tau_ext_try', tau_ext_try)

    #print('tau_no_forces', tau_ext)
    # print('tau + tau_ext', tau + tau_ext)
    # print('tau_tot', tau_tot)

    #qdd = np.linalg.solve(M, rhs) if np.linalg.cond(M) < 1e12 else np.linalg.pinv(M) @ rhs
    eps = 1e-8            # damping
        
    #rhs = tau + tau_ext - tau_prime ********


    #qdd = np.linalg.solve(M + eps*np.eye(M.shape[0]), rhs)
    #qd = qd + qdd * DT
    #q = q + qd * DT
    q, qd = rk4_step(q, qd, tau, tau_ext, tau_prime, M, DT)

    J = robot.fkine_all(q)
    tau_ext_th = J

    # momentum residuals 
    res, p_hat = residuals.momentum_residuals(robot, q, qd, tau, tau_prime, M, M_dot, K_0, p_hat, res, DT)
    #print('residuals: ', res)

    # SOLVE THE SYSTEM 
    # SINGLE CASE

    knowns = {
    solver.F_tot_x: Fx_tot,     # known total force x
    solver.F_tot_y: Fy_tot,      # known total force y
    solver.q1: q[0],      # orientation
    solver.q2: q[1],
    solver.q3: q[2],
    solver.q4: q[3],
    solver.l1 : link_lengths[0],
    solver.l2 : link_lengths[1],
    solver.l3 : link_lengths[2],
    solver.l4 : link_lengths[3]
    }

    # Add taus only if conditions are met
    if case_single >= 1:
        knowns[solver.tau_1] = res[0]
    if case_single >= 2:
        knowns[solver.tau_2] = res[1]
    if case_single >= 3:
        knowns[solver.tau_3] = res[2]
    if case_single >= 4:
        knowns[solver.tau_4] = res[3]

    print('q1', q[0])
    solution = solver.solve(knowns)
    print('solution: ', solution)
    if solution:  # se esiste almeno una soluzione
        sol = solution[0]
        F_x_val = sol.get(solver.F_x, None)
        F_y_val = sol.get(solver.F_y, None)
        l_val   = sol.get(solver.l_bar, None)
        F_x_log[k], F_y_log[k], l_log[k] = F_x_val, F_y_val, l_val
    # assegni ai tuoi log
    #F_x_log[k], F_y_log[k], l_log[k] = F_x_val, F_y_val, l_val
    # Log
    q_log[k], qd_log[k], tau_log[k], tau_prime_log[k], res_log[k], tau_ext_log[k] = q, qd, tau, tau_prime, res, tau_ext

    if ANIMATE:
        update_custom_3d(ax, robot, q, reach = 2, margin=.4)



# -------------------------
# Plots
# --------------------------
labels_q = [f"q{i+1}" for i in range(n)]
labels_qd = [f"q̇{i+1}" for i in range(n)]


plt.figure()
plt.plot(time, q_log[:, :3])
plt.title("Anthro 3R Joint Angles")
plt.xlabel("Time [s]")
plt.ylabel("q [rad]")
plt.legend(labels_q)

plt.figure()
plt.plot(time, qd_log)
plt.title("Anthro 3R Joint Velocities")
plt.xlabel("Time [s]")
plt.ylabel("q̇ [rad/s]")
plt.legend(labels_qd)

# plt.figure()
# plt.plot(time, tau_log)
# plt.title("Applied Joint Torques")
# plt.xlabel("Time [s]")
# plt.ylabel("τ [Nm]")
# plt.legend([f"τ{i+1}" for i in range(n)])

# plt.figure()
# plt.plot(time, tau_prime_log)
# plt.title("tau_prime")
# plt.xlabel("Time [s]")
# plt.ylabel("tau_prime [Nm]")
# plt.legend([f"tau_prime{i+1}" for i in range(n)])

plt.figure()
plt.plot(time, res_log)
plt.title("Momentum Residuals")
plt.xlabel("Time [s]")
plt.ylabel("res [Nm]")
plt.legend([f"res{i+1}" for i in range(n)])

plt.figure()
plt.plot(time, tau_ext_log)
plt.title("tau_ext")
plt.xlabel("Time [s]")
plt.ylabel("tau_ext [Nm]")
plt.legend([f"tau_ext{i+1}" for i in range(n)])

plt.figure()
plt.plot(time, F_x_log)
plt.title("F_x")
plt.xlabel("Time [s]")
plt.ylabel("F_x [Nm]")
plt.legend([f"F_x{i+1}" for i in range(n)])

plt.figure()
plt.plot(time, F_y_log)
plt.title("F_y")
plt.xlabel("Time [s]")
plt.ylabel("F_y [Nm]")
plt.legend([f"F_y{i+1}" for i in range(n)])

plt.figure()
plt.plot(time, l_log)
plt.title("l")
plt.xlabel("Time [s]")
plt.ylabel("l [Nm]")
plt.legend([f"l{i+1}" for i in range(n)])

plt.show()
plt.show(block=False)
plt.pause(4) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run