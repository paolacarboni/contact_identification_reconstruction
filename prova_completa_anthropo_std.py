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


### DEBUG 
print('started anthropomorphic arm')
print(frne.ping())

print('check until here')
# --------------------------
# Apply custom style (optional)
# --------------------------
plt.style.use("ggplot")  # change to your custom .mplstyle if desired

# --------------------------
# Anthropomorphic 3R Arm with inertia
# --------------------------
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class AnthropomorphicArm:
    def __init__(self, link_lengths, link_masses, link_radiuses):
        self.l = link_lengths
        self.m = link_masses
        self.r = link_radiuses
        self.robot = self._create_robot()

    def _create_robot(self):
        L = []

        # Link 1: vertical offset (base)
        r1 = [0, 0, -self.l[0] / 2]
        #Izz1 = self.m[0] * (self.l[0] ** 2) / 12
        Ixy = 1/12*(3*self.r[0]**2+self.l[0]**2)
        Iz = 1/2*self.l[0]**2
        I1 = self.m[0]*np.diag([Ixy, Ixy, Iz])

        #I1 = np.diag([0, 0, Izz1])
        L1 = RevoluteDH(
            a=0, alpha=np.pi/2, d=0,
            m=self.m[0], r=r1, I=I1, Jm=0, G=1
        )

        # Link 2: shoulder rotation (rotates around z, offset in y due to alpha=pi/2)
        r2 = [-self.l[1] / 2, 0, 0]
        #Izz2 = self.m[1] * (self.l[1] ** 2) / 12

        Ix = 1/2*self.r[1]**2
        Iyz = 1/12*(3*self.r[1]**2+self.l[1]**2)
        I2 = self.m[1]*np.diag([Ix, Iyz, Iyz])

        #I2 = np.diag([0, 0, Izz2])
        L2 = RevoluteDH(
            a=self.l[1], alpha=0, d=0,
            m=self.m[1], r=r2, I=I2, Jm=0, G=1
        )

        # Link 3: elbow rotation
        r3 = [-self.l[2] / 2, 0, 0]
        #Izz3 = self.m[2] * (self.l[2] ** 2) / 12
        Ix = 1/2*self.r[2]**2
        Iyz = 1/12*(3*self.r[2]**2+self.l[2]**2)
        I3 = self.m[2]*np.diag([Ix, Iyz, Iyz])
        #I3 = np.diag([0, 0, Izz3])
        L3 = RevoluteDH(
            a=self.l[2], alpha=0, d=0,
            m=self.m[2], r=r3, I=I2, Jm=0, G=1
        )

        # Link 4: wrist (dummy for EE position)
        # L4 = RevoluteMDH(a=self.l[2], alpha=0, d=0)

        return my_DHRobot_fext([L1, L2, L3], name='Anthro3R')

    def plot(self, q=None, block=True):
        if q is None:
            q = [0, 0, 0]
        self.robot.plot(q, block=block)

# --------------------------
# Instantiate robot
# --------------------------
link_lengths = [0.5, 0.5, 0.5]   # l1, l2, l3
#link_masses  = [0.7, 0.7, 0.7]         # m1, m2, m3
link_masses = [15, 10, 5]
link_radiuses = [0.2, 0.1, 0.1]

my_robot = AnthropomorphicArm(link_lengths, link_masses, link_radiuses)
robot = my_robot.robot
n = robot.n # number of joints

# Suppose n joints
fext_links = [[0,0,0, 0,0,0] for _ in range(n)] # Fx Fy Fz Mx My Mz
pext_links = [[0,0,0] for _ in range(n)] # px py pz

# Apply a 10 N force along +x at 0.2 m along link 3 (index 2), no moment
fext_links[2] = [0,0,0, 0,0,0]
pext_links[2] = [0.4,0,0]
# --------------------------
# Simulation config
# --------------------------
#DT = 0.002
DT = 0.002
T  = 5.0
Kp = 30 * np.diag([1, 1, 1])
Kd = 30 * np.diag([1, 1, 1])
K_0 = 20 * np.diag([1, 1, 1]) # residual gain

q0 = np.array([0, 0, 0])
qf = np.array([0, 0, 0])

time = np.arange(0, T, DT)
N = len(time)

#gravity
g_0 = np.array([0, 0, -9.81])
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
res_log = np.zeros((N, n))

F_ext_ee = np.array([0,0,-30, 0, 0, 0])

fext_links = np.zeros((n, 3), dtype=np.float64)
fext_links_zeros = np.zeros((n, 3), dtype=np.float64)
mext_links = np.zeros((n, 3), dtype=np.float64)
pext_links = np.zeros((n, 3), dtype=np.float64)
res = np.zeros((n), dtype=np.float64)
p_hat = np.zeros((n), dtype=np.float64)

bool_f2 = True
bool_f3 = False

F2_ext = np.array([0,-50, 0])
P2_ext = np.array([-0.1,0,0])

F3_ext = np.array([0,-50,0])
P3_ext = np.array([-0.1,0,0])

#time_interval_1 = np.array([0, 0.5])
time_interval_2 = np.array([2.5, 3.5])
time_interval_3 = np.array([2.0, 2.5])

# --------------------------
# 3D animation helpers
# --------------------------
def init_custom_3d():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
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

def update_custom_3d(ax, robot, q):
    ax.cla()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
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
        draw_frame(ax, T)
    ax.plot(xs, ys, zs, '-o', color='royalblue', markersize=4, linewidth=2)
    ax.scatter(xs[-1], ys[-1], zs[-1], color='red', s=80, edgecolors='k')
    plt.pause(0.001)

def rk4_step(q, qd, tau, tau_prime, M, DT):
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
        qdd = np.linalg.pinv(M) @ (tau - tau_prime)
        return np.concatenate([qd, qdd])

    y = np.concatenate([q, qd])

    k1 = f(y)
    k2 = f(y + 0.5 * DT * k1)
    k3 = f(y + 0.5 * DT * k2)
    k4 = f(y + DT * k3)

    y_new = y + DT/6 * (k1 + 2*k2 + 2*k3 + k4)

    return y_new[:n], y_new[n:]

# --------------------------
# Simulation loop
# --------------------------
ANIMATE = False
if ANIMATE:
    fig, ax = init_custom_3d()

for k, t in enumerate(time):
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
    tau_prime = robot.rne(q, qd, np.zeros(n), gravity = g_0, fext_links = fext_links, pext_links = pext_links)#, ext_forces=Fe, ext_moments=Ne, ext_points=Re)

    #rhs = tau - C @ qd - g

    #qdd = np.linalg.solve(M, rhs) if np.linalg.cond(M) < 1e12 else np.linalg.pinv(M) @ rhs
    eps = 1e-8            # damping
    rhs = tau - tau_prime
    qdd = np.linalg.solve(M + eps*np.eye(M.shape[0]), rhs)
    #qd = qd + qdd * DT
    #q = q + qd * DT
    q, qd = rk4_step(q, qd, tau, tau_prime, M, DT)

    # momentum residuals 
    res, p_hat = residuals.momentum_residuals(robot, q, qd, tau, tau_prime, M, M_dot, K_0, p_hat, res, DT)
    #print('residuals: ', res)

    # Log
    q_log[k], qd_log[k], tau_log[k], tau_prime_log[k], res_log[k] = q, qd, tau, tau_prime, res

    if ANIMATE:
        update_custom_3d(ax, robot, q)

# --------------------------
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

plt.figure()
plt.plot(time, tau_log)
plt.title("Applied Joint Torques")
plt.xlabel("Time [s]")
plt.ylabel("τ [Nm]")
plt.legend([f"τ{i+1}" for i in range(n)])

plt.figure()
plt.plot(time, tau_prime_log)
plt.title("tau_prime")
plt.xlabel("Time [s]")
plt.ylabel("tau_prime [Nm]")
plt.legend([f"tau_prime{i+1}" for i in range(n)])

plt.figure()
plt.plot(time, res_log)
plt.title("Momentum Residuals")
plt.xlabel("Time [s]")
plt.ylabel("res [Nm]")
plt.legend([f"res{i+1}" for i in range(n)])

plt.show()
