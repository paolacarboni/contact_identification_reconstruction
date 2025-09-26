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
import time
from mpl_toolkits.mplot3d.art3d import Line3D
import os 

path =  '/home/pcarboni/MARR/TESI/GRAFICI/PLANAR/'

# set global plot style
plt.style.use("seaborn-whitegrid")  # ✅ same style as your Anthro3R
plt.rcParams["lines.linewidth"] = 2
#plt.style.use("default") 

# Change font sizes globally
plt.rcParams["axes.titlesize"] = 14   # title font
plt.rcParams["axes.labelsize"] = 12   # x and y axis labels
plt.rcParams["axes.titleweight"] = "bold"  # title weight
plt.rcParams["xtick.labelsize"] = 10  # x tick labels
plt.rcParams["ytick.labelsize"] = 10  # y tick labels
plt.rcParams["legend.fontsize"] = 10   # 👈 legend font size

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
    ax.set_zlim([-3, 3])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Anthropomorphic 3R Arm")
    return fig, ax

def init_custom_2d():
    fig, ax = plt.subplots(figsize=(6, 6))
    return fig, ax

# def draw_frame(ax, T, length=0.2):
#     origin = T.t
#     R = T.R
#     ax.quiver(*origin, *R[:, 0] * length, color='r', linewidth=1.2)
#     ax.quiver(*origin, *R[:, 1] * length, color='g', linewidth=1.2)
#     ax.quiver(*origin, *R[:, 2] * length, color='b', linewidth=1.2)
#     set_axes_equal(ax)

######
# def draw_arrow(ax, start, vec, color='r', lw=2):
#     end = start + vec
#     line = Line3D([start[0], end[0]],
#                   [start[1], end[1]],
#                   [start[2], end[2]],
#                   color=color, linewidth=lw)
#     ax.add_line(line)

# def draw_frame_fixed(ax, T, length=0.2):
#     origin = T.t
#     R = T.R

#     for i, color in zip(range(3), ['r','g','b']):
#         vec = R[:, i] * length/[1, 1, 35]
#         draw_arrow(ax, origin, vec, color=color)

        ########

def draw_arrow(ax, start, vec, color='r', lw=2):
    end = start + vec
    line = Line3D([start[0], end[0]],
                  [start[1], end[1]],
                  [start[2], end[2]],
                  color=color, linewidth=lw)
    ax.add_line(line)

# def draw_frame(ax, T, scale = 0.05):
#     origin = T.t
#     R = T.R
#     length = scale
#     for i, color in zip(range(3), ['r', 'g', 'b']):
#         vec = R[:, i] * length
#         draw_arrow(ax, origin, vec, color=color)
#         # vec = R[:, i] / np.linalg.norm(R[:, i]) * length
#         # draw_arrow(ax, origin, vec, color=color)
#         # 

def draw_frame(ax, T, length=0.01):
    origin = T.t
    R = T.R

    # Get axis ranges
    x_range = ax.get_xlim3d()[1] - ax.get_xlim3d()[0]
    y_range = ax.get_ylim3d()[1] - ax.get_ylim3d()[0]
    z_range = ax.get_zlim3d()[1] - ax.get_zlim3d()[0]

    max_range = max(x_range, y_range, z_range)

    # Scale uniformly based on max range
    scale = length / max_range

    scaling_factors = np.array([x_range, y_range, z_range]) * length # normalizzazione per avere tutte frecce della stessa lunghezza
    #scaling_factors = np.array([0.5, 0.5, 0.01])
    for i, color in zip(range(3), ['r', 'g', 'b']):
        vec = R[:, i] * scaling_factors
        #draw_arrow(ax, origin, vec, color=color) 
        draw_arrow_with_head(ax, origin, vec, color=color)

def draw_frame_2d(ax, T, length=None):
    Tmat = T.A  # Convert SE3 to NumPy array
    origin = T.t #Tmat[0:2, 3]
    # print('origin: ', origin)
    R = T.R #Tmat[0:2, 0:2]
    # print('R: ', R)

    # Get axis ranges for visual scaling
    # x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    # y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    # max_range = max(x_range, y_range)
    # scale = length / max_range
    scale = 0.3
    x_axis = R[:, 0] * scale
    y_axis = R[:, 1] * scale
    # ax.arrow(origin[0], origin[1], x_axis[0], x_axis[1],
    #          head_width=0.02, color='r', length_includes_head=True)
    # ax.arrow(origin[0], origin[1], y_axis[0], y_axis[1],
    #          head_width=0.02, color='g', length_includes_head=True)
    draw_arrow_with_head_2d(ax, origin, x_axis, color='r')
    draw_arrow_with_head_2d(ax, origin, y_axis, color='g')

def draw_arrow_with_head(ax, start, vec, color='r', lw=1.5, arrow_length_ratio=0.05):
    ax.quiver(*start, *vec, color=color, linewidth=lw, arrow_length_ratio=arrow_length_ratio, normalize=False) 

def draw_arrow_with_head_2d(ax, start, vec, color='r', lw=1.5, arrow_length_ratio=0.05):
    # print('start: ', start)
    # print('vec: ', vec)
    ax.quiver(start[0], start[1], vec[0], vec[1],
              color=color, linewidth=lw,
              angles='xy', scale_units='xy', scale=1,
              width=0.005, headwidth=3, headlength=5)
    
def draw_force(ax, force_base_frame, T, end, num_forces, case, index, scale=0.1, color='#8a1c7c'):
    # Convert force to numpy array
    #force = np.array(force_base_frame)

    # # Normalize and scale the force vector
    # if np.linalg.norm(force) != 0:
    #     force_scaled = force / np.linalg.norm(force) * scale
    # else:
    #     force_scaled = np.zeros_like(force)

    # # Extract position from transformation matrix T
    # pos = T.A[0:2, 3] if hasattr(T, 'A') else np.array(position)
    if num_forces ==1:
        force_base_frame = force_base_frame/np.linalg.norm(force_base_frame)*0.5
    else:
        force_base_frame = -force_base_frame / 250
    #force_base_frame = -force_base_frame / 50000*np.linalg.norm(force_base_frame)
    # Draw the force arrow
    # print('T: ', type(T), T)
    end = [end[0], end[1], end[2], 1]
    T = T.A
    T = np.array(T)
    end = np.array(end)
    end = T @ end
    pos = end[:3] - force_base_frame
    ax.quiver(pos[0], pos[1], force_base_frame[0], force_base_frame[1],
              color=color, angles='xy', scale_units='xy', scale=1,
              width=0.005, headwidth=3, headlength=5)
    
     # Place label near the tail
    if not np.all(force_base_frame == 0):
        offset = np.array([-0.02, -0.25])  # tweak for spacing
        label_pos = pos[:2] + offset
        if num_forces == 1:
            label = "$F$"
        if num_forces == 2:
            if index == 0 or index == 1 or index == 2:
                label = "$F_A$"
            if index == 3:
                label = "$F_B$"
        ax.text(label_pos[0], label_pos[1], label,
                color=color, fontsize=12, ha='right', va='bottom')

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

link_lengths = [0.5, 0.5, 0.5, 0.5]
link_masses  = [5, 5, 5, 5]
link_radiuses = [0.1, 0.1, 0.1, 0.1]

p4r = Planar_4r(link_lengths, link_masses, link_radiuses)
robot = p4r.robot

# Setup figure
#fig, ax = init_custom_3d()
fig, ax = init_custom_2d()

# # Animate a trajectory
# q_traj = np.linspace(0, np.pi/2, 200)   # simple joint motion
# for q1 in q_traj:
#     q = [q1, -q1/2, q1/3, -q1/4]
#     update_custom_3d(ax, robot, q)


# def draw_frame(ax, T, reach, upper_limit, length=0.1):
#     origin = T.t
#     R = T.R
#     ax.quiver(*origin, *R[:, 0] * length*reach, color="r", linewidth=1.2, arrow_length_ratio=0.02)
#     ax.quiver(*origin, *R[:, 1] * length*reach, color="g", linewidth=1.2, arrow_length_ratio=0.02)
#     ax.quiver(*origin, *R[:, 2] * length/6, color="b", linewidth=1.2, arrow_length_ratio=0.7)


def update_custom_3d(ax, robot, q, reach, margin, time_value = None):
    upper_limit = 0.2
    ax.cla()
    ax.set_xlim([-reach - margin, reach + margin])
    ax.set_ylim([-reach - margin, reach + margin])
    ax.set_zlim([0, upper_limit])
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
        draw_frame(ax, T, length = 0.05)
        #draw_force(ax, p, T.c, scale=0.1, color="magenta")
    ax.plot(xs, ys, zs, "-o", color="royalblue", markersize=6, linewidth=2)
    ax.scatter(xs[-1], ys[-1], zs[-1], color="red", s=80, edgecolors="k")
    if time_value is not None:
        ax.text2D(0.05, 0.95, f"Time: {time_value:.2f} s", transform=ax.transAxes, fontsize=15, color='black')
    #set_axes_equal(ax)
    plt.pause(0.001)

def update_custom_2d(ax, robot, q, forces_base_frame, contact_points_link_frame, num_forces, case, reach=2, margin=0.4, time_value=None):
    ax.cla()
    ax.set_xlim([-reach - margin, reach + margin])
    ax.set_ylim([-reach - margin, reach + margin])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Planar 4R Arm")

    Ts = robot.fkine_all(q)
    xs, ys = [], []
    i = 0
    for T in Ts:
        p = T.t
        xs.append(p[0])
        ys.append(p[1])
        draw_frame_2d(ax, T, length=0.2)
        if i < len(forces_base_frame):
            draw_force(ax, forces_base_frame[i], Ts[i+1], end=contact_points_link_frame[i], num_forces=num_forces, case=case, index=i, scale=0.1, color="magenta")
        if i == 0: 
            offset = np.array([-0.2, 0.25])  # tweak for spacing
            label_pos = p[:2] - offset
            label = "$RF_0$"
            ax.text(label_pos[0], label_pos[1], label,
                color='b', fontsize=12, ha='right', va='bottom')
        if i == 4: 
            offset = np.array([0.25, 0.1])  # tweak for spacing
            label_pos = p[:2] + offset
            label = "$ee$"
            ax.text(label_pos[0], label_pos[1], label,
                color='r', fontsize=12, ha='right', va='bottom')
        i += 1
                

    #points = np.column_stack((xs, ys))
    ax.plot(xs, ys, "-o", color="royalblue", linewidth=2, markersize=6) #ax.plot(points[:, 0], points[:, 1], "-o", color="royalblue", linewidth=2, markersize=6)
    ax.scatter(xs[-1], ys[-1], color="red", s=80, edgecolors="k") #ax.scatter(points[-1, 0], points[-1, 1], color="red", s=80, edgecolors="k")

    if time_value is not None:
        ax.text(0.05, 0.95, f"Time: {time_value:.2f} s", transform=ax.transAxes, fontsize=12, color='black')

    ax.grid(True)
    ax.set_aspect("equal")
    plt.pause(0.001)

def get_base_to_link_transformation(robot, q):
    Ts = robot.fkine_all(q)
    return Ts

print('get_base_to_link_transformation: ', get_base_to_link_transformation(robot, [0,0,0,0]))

# --------------- Run animation -------------- #
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
T  = 10.0
# Kp = 30 * np.diag([1, 1, 1,1 ])
# Kd = 30 * np.diag([1, 1, 1, 1])

Kp = 0 * np.diag([1, 50, 50,50 ])
Kd = 0 * np.diag([1, 50, 50, 50])


K_0 = 20 * np.diag([1, 1, 1, 1]) # residual gain

# pentagono
# q0 = np.array([np.pi/3, np.pi/2, np.pi/3, -3*np.pi/2])
# qf = np.array([np.pi/3, np.pi/2, np.pi/3, -3*np.pi/2])

#q0 = np.array([pi/2, -pi/2, pi/2, -pi/2])

q0 = np.array([0, 0, 0, 0])
qf = np.array([0, 0, 0, 0])

# scaletta
# q0 = np.array([0, np.pi/2, -np.pi/2, np.pi/2])
# qf = np.array([np.pi/2, 0, 0, 0])

# s shape
# q0 = np.array([np.pi/4, -np.pi/4, -np.pi/3, np.pi/3])
# qf = np.array([np.pi/4, -np.pi/4, -np.pi/3, np.pi/3])

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
Fx_tot_log = np.zeros(N)
Fy_tot_log = np.zeros(N)
Fa_x_gt_log = np.zeros(N)
Fa_y_gt_log = np.zeros(N)
la_bar_gt_log = np.zeros(N)
Fb_x_gt_log = np.zeros(N)
Fb_y_gt_log = np.zeros(N)
lb_bar_gt_log = np.zeros(N)

# rotation matrices 
R = np.zeros((n, 3, 3))

# real values for single contact force
F_x_real_log = np.zeros(N)
F_y_real_log = np.zeros(N)
l_contact_log = np.zeros(N)

# reconstructed values, single contact force
F_x_log = np.zeros(N)
F_y_log = np.zeros(N)
l_log = np.zeros(N)

# reconstructed values for double contact
Fa_x_log = np.zeros(N)
Fa_y_log = np.zeros(N)
la_bar_log = np.zeros(N)

Fb_x_log = np.zeros(N)
Fb_y_log = np.zeros(N)
lb_bar_log = np.zeros(N)


F_ext_ee = np.array([0,0,0, 0, 0, 0])
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
F1_ext_ = np.array([0, 400, 0])
P1_ext_ = np.array([-0.1,0,0])

F2_ext_ = np.array([0,400, 0])
P2_ext_ = np.array([-0.1,0,0])

F3_ext_ = np.array([0,200,0])
P3_ext_ = np.array([-0.1,0,0])

F4_ext_ = np.array([50, 150,0])
P4_ext_ = np.array([-0.1,0,0])   

#time_interval_1 = np.array([0, 0.5])
time_interval_1 = np.array([0.5, 1.5])
time_interval_2 = np.array([0.1, 1.5])
time_interval_3 = np.array([0.1, 0.5])
time_interval_4 = np.array([0.5, 2.0])

fext_base_array = np.zeros((n, 3), dtype=np.float64)
#case for single force applied
num_forces =  1   #[1, 2] how many external forces are applied
case_single = 2 # [1, 2, 3, 4] on which link is the force applied
case_double = 14 # [14, 24, 34, 44] on which link is the force applied

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

single_path = os.path.join(path, '1FORCE')
double_path = os.path.join(path, '2FORCES')

debug_Fa = True
debug_Fb = True
ANIMATE = True#
if ANIMATE:
    #fig, ax = init_custom_3d()
    fig, ax = init_custom_2d()

for k, t in enumerate(time):
    print('t', t)
    #print('entered enumerate time')
    q_ref, qd_ref, qdd_ref = q_d[k], qd_d[k], qdd_d[k]
    e, edot = q_ref - q, qd_ref - qd

    # initialization of the contact lengths
    l_log[k] = 0
    la_bar_log[k] = 0
    lb_bar_log[k] = 0

    # Newton–Euler feedforward + PD feedback
    #tau_ff = robot.rne(q_ref, qd_ref, qdd_ref, gravity = g_0, ext_forces=Fe, ext_moments=Ne, ext_points=Re)
    # print(' tau ff: ', tau_ff)
    
    tau_fb = Kp @ e + Kd @ edot
    #tau = tau_ff + tau_fb
    tau = tau_fb

    # external forces
    l_contact = 0
    la_bar_gt = 0
    lb_bar_gt = 0
    F_x_real = 0
    F_y_real = 0
    Fa_x_gt = np.float64(0)
    Fa_y_gt = 0
    Fb_x_gt = 0
    Fb_y_gt = 0

    if bool_f1 and t > time_interval_1[0] and t < time_interval_1[1]:
        fext_links[0] = F1_ext
        pext_links[0] = P1_ext
        l_contact = P1_ext[0]
        if num_forces == 1:
            F_x_real = F1_ext[0]
            F_y_real = F1_ext[1]
            print('F_y_real', F_y_real)
        if num_forces == 2 and debug_Fa == True:
            Fa_x_gt = F1_ext[0]
            print('if bool Fa_x_gt', Fa_x_gt)
            print(type(Fa_x_gt))
            Fa_y_gt = F1_ext[1]
            la_bar_gt = l_contact
    else:
        fext_links[0] = np.zeros(3)
        pext_links[0] = np.zeros(3)
        #bool_f1 = False
    if bool_f2 and t > time_interval_2[0] and t < time_interval_2[1]:
        fext_links[1] = F2_ext
        pext_links[1] = P2_ext
        if num_forces == 1:
            F_x_real = F2_ext[0]
            F_y_real = F2_ext[1]
        if num_forces == 2 and debug_Fa == True:
            Fa_x_gt = F2_ext[0]
            Fa_y_gt = F2_ext[1]
            la_bar_gt = l_contact
    else:
        fext_links[1] = np.zeros(3)
        pext_links[1] = np.zeros(3)
        #bool_f2 = False
    if bool_f3 and t > time_interval_3[0] and t < time_interval_3[1]:
        fext_links[2] = F3_ext
        pext_links[2] = P3_ext
        if num_forces == 1:
            F_x_real = F3_ext[0]
            F_y_real = F3_ext[1]
        if num_forces == 2 and debug_Fa == True:
            Fa_x_gt = F3_ext[0]
            Fa_y_gt = F3_ext[1]
            la_bar_gt = l_contact
    else:
        fext_links[2] = np.zeros(3)
        pext_links[2] = np.zeros(3)
    if bool_f4 and t > time_interval_4[0] and t < time_interval_4[1]:
        fext_links[3] = F4_ext
        pext_links[3] = P4_ext
        if num_forces == 1:
            F_x_real = F4_ext[0]
            F_y_real = F4_ext[1]
        if num_forces == 2 and debug_Fb == True:
            Fb_x_gt = F4_ext[0]
            Fb_y_gt = F4_ext[1]
            lb_bar_gt = P4_ext[0]
    else:
        fext_links[3] = np.zeros(3)
        pext_links[3] = np.zeros(3)

    fext_array = np.array(fext_links)
    # print('fext_array', fext_array)
    # totals = fext_array.sum(axis=0)
    # Fx_tot = totals[0]
    # Fy_tot = totals[1]



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
    tau_ext = (tau_tot - tau_no_forces)
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

    # momentum residuals 
    res, p_hat = residuals.momentum_residuals(robot, q, qd, tau, tau_prime, M, M_dot, K_0, p_hat, res, DT)
    #print('residuals: ', res)

     # transform the forces in the base frame 
    Ts = get_base_to_link_transformation(robot, q)

    for i in range(n):
        # print('i', i)
        T_i = Ts[i+1]          # omogenea base -> link
        R[i] = T_i.R  # matrice di rotazione
        f_link = fext_array[i]
        f_base = R[i] @ f_link
        fext_base_array[i] = f_base

    totals = fext_base_array.sum(axis=0)
    # forze totali espresse nel SDR della base del robot
    Fx_tot = totals[0] 
    Fy_tot = totals[1]
    #print('Fx_tot_base: ', Fx_tot)
    #print('Fy_tot_base: ', Fy_tot)


    solve = True # flag to solve the system or not
    if solve == True:
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
        if num_forces == 1 and case_single >= 1:
            knowns[solver.tau_1] = res[0]
        if num_forces == 1 and case_single >= 2:
            knowns[solver.tau_2] = res[1]
        if num_forces == 1 and case_single >= 3:
            knowns[solver.tau_3] = res[2]
        if num_forces == 1 and case_single >= 4:
            knowns[solver.tau_4] = res[3]
        if num_forces == 2:
            knowns[solver.tau_1] = res[0]
            knowns[solver.tau_2] = res[1]
            knowns[solver.tau_3] = res[2]
            knowns[solver.tau_4] = res[3]
            print('res[0]', res[0])
            print(type(res[0]))
            # if debug_Fa == True: 
            #knowns[solver.Fa_x] = Fa_x_gt
            print('Fa_x_gt', Fa_x_gt)
            print(type(Fa_x_gt))
            #     knowns[solver.Fa_y] = Fa_y_gt
            # if debug_Fb == True:
            #     knowns[solver.Fb_x] = Fb_x_gt
            #     knowns[solver.Fb_y] = Fb_y_gt



        #elif case == 'ca'

        # #print('q1', q[0])
        if num_forces == 1:
            solution = solver.solve(knowns)
            #path = os.path.join(path, '1FORCE')
            #os.makedirs(path, exist_ok=True)
            if solution:
                sol = solution[0]
                print('solution: ', solution)
                F_sol_base = [sol.get(solver.F_x), sol.get(solver.F_y), 0] # ho trovato la forza nel frame della base
                if case == 1: 
                    # esprimo la forza nel frame dei link
                    F_sol_link = R[0].T @ F_sol_base
                    path = os.path.join(single_path, 'LINK1')
                    os.makedirs(path, exist_ok=True)
                elif case == 2:
                    F_sol_link = R[1].T @ F_sol_base
                    path = os.path.join(single_path, 'LINK2')
                    os.makedirs(path, exist_ok=True)
                elif case == 3:
                    F_sol_link = R[2].T @ F_sol_base
                    path = os.path.join(single_path, 'LINK3')
                    os.makedirs(path, exist_ok=True)
                elif case == 4:
                    F_sol_link = R[3].T @ F_sol_base
                    path = os.path.join(single_path, 'LINK4')
                    os.makedirs(path, exist_ok=True)
        elif num_forces == 2 and case == 14:
            path = os.path.join(double_path, '14') #double_path
            os.makedirs(path, exist_ok=True)
            solution = solver.block_solve_14(knowns)
            sol = solution[0]
            print('solution: ', solution)
            Fa_sol_base = [sol.get(solver.Fa_x), sol.get(solver.Fa_y), 0] # ho trovato la forza nel frame della base
            Fb_sol_base = [sol.get(solver.Fb_x), sol.get(solver.Fb_y), 0]
            la_bar_base = sol.get(solver.la_bar)
            lb_bar_base = sol.get(solver.lb_bar)
            if case == 14:
                # esprimo le forze nel frame dei link
                Fa_sol_link = R[0].T @ Fa_sol_base
                #Fa_sol_link_0 = 
                Fb_sol_link = R[3].T @ Fb_sol_base

            #print('Fa_sol_link: ', Fa_sol_link)
            #print('Fb_sol_link: ', Fb_sol_link)

       # print('solution: ', solution)
        if solution:  # se esiste almeno una soluzione
            sol = solution[0]
            if num_forces == 1:
                sol = solution[0]
                F_x_val = F_sol_link[0] #sol.get(solver.F_x, 0)
                F_y_val = F_sol_link[1] #sol.get(solver.F_y, None)
                l_val   = sol.get(solver.l_bar, 0)
                F_x_log[k], F_y_log[k], l_log[k] = F_x_val, F_y_val, l_val
            if num_forces == 2:
                #Fa_x_val = sol.get(solver.Fa_x, None)
                #Fa_y_val = sol.get(solver.Fa_y, None)
                #Fb_x_val = sol.get(solver.Fb_x, None)
                #Fb_y_val = sol.get(solver.Fb_y, None)
                Fa_x_val = Fa_sol_link[0]
                Fa_y_val = Fa_sol_link[1]
                Fb_x_val = Fb_sol_link[0]
                Fb_y_val = Fb_sol_link[1]
                la_bar_val   = sol.get(solver.la_bar, 0)
                lb_bar_val   = sol.get(solver.lb_bar, 0)
                # print('Fa_x_val : ', Fa_x_val)
                #Fa_x_log[k] = Fa_x_val
                Fa_x_log[k], Fa_y_log[k], Fb_x_log[k], Fb_y_log[k], la_bar_log[k], lb_bar_log[k] = Fa_x_val, Fa_y_val, Fb_x_val, Fb_y_val, la_bar_val, lb_bar_val
    Fa_x_gt_log[k], Fa_y_gt_log[k], Fb_x_gt_log[k], Fb_y_gt_log[k], la_bar_gt_log[k], lb_bar_gt_log[k] = Fa_x_gt, Fa_y_gt, Fb_x_gt, Fb_y_gt, la_bar_gt, lb_bar_gt
                #Fb_x_log[k], Fb_y_log[k], la_bar_log[k], lb_bar_log[k] =  Fb_x_val, Fb_y_val, la_bar_val, lb_bar_val

    print('la bar log: ', la_bar_log[k])
    # assegni ai tuoi log
    #F_x_log[k], F_y_log[k], l_log[k] = F_x_val, F_y_val, l_val
    # Log

    # groud truth value for single contact
    # print('final debugF_y_real: ', F_y_real_log[k])
    F_x_real_log[k], F_y_real_log[k], l_contact_log[k] = F_x_real, F_y_real, l_contact
    #l_contact_a_log[k] = 
    q_log[k], qd_log[k], tau_log[k], tau_prime_log[k], res_log[k], tau_ext_log[k], Fx_tot_log[k], Fy_tot_log[k] = q, qd, tau, tau_prime, res, tau_ext, Fx_tot, Fy_tot

    if ANIMATE:
        #update_custom_3d(ax, robot, q, reach = 2, margin=.4, time_value = t)
        print('f_base: ', f_base)
        update_custom_2d(ax, robot, q, fext_base_array, contact_points_link_frame=pext_links, num_forces=num_forces, case=case, reach=2, margin=0.4, time_value=t)
        #ax.text(0.5, 0.5, 0.5 + 0.2, f"Time: {t:.2f} s", fontsize=12, color='black')


# -------------------------
# Plots
# --------------------------
labels_q = [f"q{i+1}" for i in range(n)]
labels_qd = [f"dq{i+1}" for i in range(n)]
l_range = [-0.5, 0.5]

plt.figure(facecolor='white')
# ax = fig.add_subplot(111)
# ax.set_facecolor("white")
plt.plot(time, q_log)
plt.title("Joint Positions")
plt.xlabel("Time [s]")
plt.ylabel("q [rad]")
plt.legend(labels_q)
filename = 'q' + '.png'
plt.savefig(os.path.join(path, filename))
ax.grid(True)

plt.figure(facecolor='white')
# ax = fig.add_subplot(111)
# ax.set_facecolor("white")
plt.plot(time, qd_log)
plt.title("Joint Velocities")
plt.xlabel("Time [s]")
plt.ylabel("dq [rad/s]")
plt.legend(labels_qd)
filename = 'qd' + '.png'
plt.savefig(os.path.join(path, filename))
ax.grid(True)

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

plt.figure(facecolor='white')
# ax = fig.add_subplot(111)
# ax.set_facecolor("white")
plt.plot(time, res_log)
plt.title("Momentum Residuals")
plt.xlabel("Time [s]")
plt.ylabel("r [Nm]")
plt.legend([f"res{i+1}" for i in range(n)])
filename = 'residuals' + '.png'
plt.savefig(os.path.join(path, filename))
ax.grid(True)

plt.figure(facecolor='white')
# ax = fig.add_subplot(111)
# ax.set_facecolor("white")
plt.plot(time, tau_ext_log)
plt.title("Real External Torques")
plt.xlabel("Time [s]")
plt.ylabel(r"$\tau_{ext}$ [Nm]")
plt.legend([fr"$\tau_{{ext,{i+1}}}$" for i in range(n)])
filename = 'tau_ext' + '.png'
plt.savefig(os.path.join(path, filename))
ax.grid(True)

if num_forces == 1:
    y_range = [-450, 50]
    plt.figure()
    plt.plot(time, -F_x_log)
    plt.ylim(y_range)
    plt.title("Reconstructed external force $F_x$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    #plt.legend([f"F_x{i+1}" for i in range(n)])
    filename = 'Fx_reconstructed' + '.png'
    plt.savefig(os.path.join(path, filename))
    ax.grid(True)

    plt.figure()
    plt.plot(time, -F_y_log)
    plt.ylim(y_range)
    plt.title("Reconstructed external force $F_y$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fy_reconstructed' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"F_y{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, l_log)
    plt.ylim(l_range)
    plt.title("Reconstructed Contact Length")
    plt.xlabel("Time [s]")
    plt.ylabel("$l$ [m]")
    filename = 'l_reconstructed' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend(["l" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, -F_x_real_log)
    plt.ylim(y_range)
    plt.title("Real external force $F_x^{real}$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fx_real' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"F_x_real{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, -F_y_real_log)
    plt.ylim(y_range)
    plt.title("Real external force $F_y^{real}$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fy_real' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"F_y_real{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, l_contact_log)
    plt.ylim(l_range)
    plt.title("Real contact length")
    plt.xlabel("Time [s]")
    plt.ylabel("l [m]")
    filename = 'l_contact_real' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"l_contact{i+1}" for i in range(n)])
    ax.grid(True)

if num_forces == 2:
    y_range = [-450, 50]
    plt.figure()
    plt.plot(time, -Fa_x_log)
    plt.ylim(y_range)
    plt.title("Reconstructed Contact Force $F_{A,x}$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fa_x_reconstructed' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"Fa_x{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, -Fa_y_log)
    plt.ylim(y_range)
    plt.title("Reconstructed Contact Force $F_{A,y}$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fa_y_reconstructed' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"Fa_y{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, -Fb_x_log)
    plt.ylim(y_range)
    plt.title("Reconstructed Contact Force $F_{B,x}$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fb_x_reconstructed' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"Fb_x{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, -Fb_y_log)
    plt.ylim(y_range)
    plt.title("Reconstructed Contact Force $F_{B,y}$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fb_y_reconstructed' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"Fb_y{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, la_bar_log)
    plt.ylim(l_range)
    plt.title(r"Reconstructed Contact Length $\bar{l}_A$")
    plt.xlabel("Time [s]")
    plt.ylabel("l [m]")
    filename = 'la_bar_reconstructed' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"la_bar{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, lb_bar_log)
    plt.ylim(l_range)
    plt.title(r"Reconstructed Contact Length $\bar{l}_B$")
    plt.xlabel("Time [s]")
    plt.ylabel("l [m]")
    filename = 'lb_bar_reconstructed' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"lb_bar{i+1}" for i in range(n)])
    ax.grid(True)

    # GT values of the forces
    plt.figure()
    plt.plot(time, -Fa_x_gt_log)
    plt.ylim(y_range)
    plt.title("Ground Truth Contact Force $F_{A,x}$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fa_x_gt' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"Fa_x_gt{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, -Fa_y_gt_log)
    plt.ylim(y_range)
    plt.title("Ground Truth Contact Force $F_{A,y}$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fa_y_gt' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"Fa_y_gt{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, -Fb_x_gt_log)
    plt.ylim(y_range)
    plt.title("Ground Truth Contact Force $F_{B,x}$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fb_x_gt' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"Fb_x_gt{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, -Fb_y_gt_log)
    plt.ylim(y_range)
    plt.title("Ground Truth Contact Force $F_{B,y}$")
    plt.xlabel("Time [s]")
    plt.ylabel("F [N]")
    filename = 'Fb_y_gt' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"Fb_y_gt{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, la_bar_gt_log)
    plt.ylim(l_range)
    plt.title(r"Ground Truth Contact Length $\bar{l}_A$")
    plt.xlabel("Time [s]")
    plt.ylabel("l [m]")
    filename = 'la_bar_gt' + '.png'
    plt.savefig(os.path.join(path, filename))
   # plt.legend([f"la_bar_gt{i+1}" for i in range(n)])
    ax.grid(True)

    plt.figure()
    plt.plot(time, lb_bar_gt_log)
    plt.ylim(l_range)
    plt.title(r"Ground Truth Contact Length $\bar{l}_B$")
    plt.xlabel("Time [s]")
    plt.ylabel("l [m]")
    filename = 'lb_bar_gt' + '.png'
    plt.savefig(os.path.join(path, filename))
    #plt.legend([f"lb_bar_gt{i+1}" for i in range(n)])
    ax.grid(True)


    # plt.figure()
    # plt.plot(time, Fx_tot_log)
    # plt.title("F_x^{2}")
    # plt.xlabel("Time [s]")
    # plt.ylabel("F_x_tot [Nm]")
    # plt.legend([f"Fx_tot{i+1}" for i in range(n)])
    # ax.grid(True)

    # plt.figure()
    # plt.plot(time, Fy_tot_log)
    # plt.title("Fy_tot")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Fy_tot [Nm]")
    # plt.legend([f"Fy_tot{i+1}" for i in range(n)])
    # ax.grid(True)


plt.show()
#plt.show(block=False)
plt.pause(4) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run