
# THIN AIRFOIL VORTEX PANEL METHOD #

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numba as nb
import warnings
from post_process import plot_circulatory_loads, compute_velocity_field_us, compute_velocity_field_ss

warnings.simplefilter('ignore', category=nb.errors.NumbaPerformanceWarning)

# ---------------------------------- #
# Flags
# ---------------------------------- #

print('Start.\n')
print('...Setting flags based on user input.')

enable_cos_dist     = False
enable_flap         = False
enable_pitching     = True
enable_gust         = True
add_meshrefinement  = False
apply_camber        = False
visualize_wake      = False

plot_backmesh       = False
plot_camber         = False
plot_ss_cl_curve    = False
plot_velocity_field = True
plot_pressure_field = False
plot_CLcirc         = True
plot_dt_comp        = False
plot_deltaP_comp    = False

# ---------------------------------- #
# Geometry and operations parameters
# ---------------------------------- #

print('...Read global parameters.')

# Airfoil
Npan = 40               # Number of panels
c = 1                   # (m) chord length
camber = 0.04           # (x/c) max camber

# Flap
c_flap = 0.3            # Fraction of chord length
a_flap = 15             # (deg) Flap deflection in degrees
Npan_flap = 10          # Number of panels on flap

# Mesh
xres = 200              # Grid discretization in x direction
yres = 50               # Grid discretization in y direction

# Operations
rho = 1.225                         # (kg/m^3) free-stream density
U_0 = 10                            # (m/s) free-stream velocity in x
v_gust = 2                          # gust in vertical direction [m/s]
AoA = 6                             # Angle of attack (for specific alpha cases)
alpha_range = np.arange(-4, 15)     # Range of AoA for cl curves
k = 0.1                             # (Hz) Reduced frequency: 0.02, 0.05, 0.1
omega = k*2*U_0/c                   # (Hz) Frequency of the unsteadiness
amp = np.deg2rad(AoA)               # (rad) Amplitude of the pitching motion

# Time
start = 0                                   # Start time
stop = 10                                   # Stop time
dt = 0.1                                    # Time step
trange = np.arange(start, stop + dt, dt)    # Time log
alpha_arr = np.zeros(len(trange))
gust_start = 5                              # gust start time [s]

# ---------------------------------- #
# Create Grid
# ---------------------------------- #

print('...Creating background grid and refinement.')

if visualize_wake:

    wl = 25         # wake length

else:

    wl = 2          # wake length

if add_meshrefinement:

    # Add refinement in the vicinity of the flat plate
    xrange = np.linspace(-1.0 * c, wl * c, xres)*(1 - 0.2 * np.sin(np.linspace(0, np.pi, xres)))
    yrange = np.linspace(-3.0 * c, 3.0 * c, yres)*(1 - 0.9 * np.sin(np.linspace(0, np.pi, yres)))

else:

    xrange = np.linspace(-1.0 * c, wl * c, xres)
    yrange = np.linspace(-3.0 * c, 3.0 * c, yres)

# Create grid
[X, Y] = np.meshgrid(xrange, yrange)

# ---------------------------------- #
# Vortex Panel Functions
# ---------------------------------- #


def naca4(c, m, p, x):  # Calculate corresponding Y components of camber line for naca 4series

    # c = chord length, m = max camber, p = max camber position, x are x coordinates

    print('...Building NACA 4series camber line.')

    x1 = x[np.where(x <= p)]    # X components upto largest thickness point
    x2 = x[np.where(x > p)]     # X components after largest thickness point

    y1 = m / p ** 2 * (2 * p * x1 - x1 ** 2)                        # y with corresponding x1
    y2 = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x2 - x2 ** 2)    # y with corresponding x2
    y = np.concatenate((y1, y2))                                    # Combined y

    return y


def indvel(gammaj, x, y, xj, yj):  # Formula 11.1

    rj_sqrt = (x - xj) ** 2 + (y - yj) ** 2
    randommatrix = np.array([[0, 1], [-1, 0]])
    xymatrix = np.array([[x - xj], [y - yj]])
    uv = gammaj / (2 * np.pi * rj_sqrt) * np.matmul(randommatrix, xymatrix)

    return uv


@nb.njit
def lumpvor2d(xcol, zcol, xvor, zvor, circvor=1):
    """
    Compute the velocity at an arbitrary collocation point (xcol, zcol) due
    to vortex element of circulation circvor, placed at (xvor, zvor).

    :param xcol: x-coordinate of the collocation point
    :param zcol: z-coordinate of the collocation point
    :param xvor: x-coordinate of the vortex
    :param zvor: z-coordinate of the vortex
    :param circvor: circulation strength of the vortex (base units)

    :return: 1D array containing the velocity vector (u, w) (x-comp., z-comp.)
    :rtype: ndarray

    """

    # transformation matrix for the x, z distance between two points
    dcm = np.array([[0.0, 1.0],
                    [-1.0, 0.0]])

    # magnitude of the distance between two points
    r_vortex_sq = (xcol - xvor) ** 2 + (zcol - zvor) ** 2

    # the distance in x, and z between two points
    dist_vec = np.array([xcol - xvor, zcol - zvor])

    norm_factor = circvor / (2.0 * np.pi * r_vortex_sq)  # circulation at
    # vortex element / circumferential distance

    # induced velocity of vortex element on collocation point
    vel_vor = norm_factor * dcm @ dist_vec

    return vel_vor


def aijmatrix(xcols, ycols, xvorts, yvorts, Npan, ni):  # Calculation of the influence coefficients

    print('   ...Computing influence matrix.')

    a_mat = np.zeros((Npan, Npan))

    print('   ...Computing induced velocities.')

    for i in range(0, Npan):
        for j in range(0, Npan):
            vel_vor = lumpvor2d(xcols[i], ycols[i], xvorts[j], yvorts[j])
            a_ij = vel_vor @ ni[:, i]
            a_mat[i, j] = a_ij

    return a_mat

@nb.njit
def aijmatrix2(a_mat, xi, yi, x_wake, y_wake, ni, wake_gamma):
    """
    Calculation of the influence coefficients

    :param xi: x-coords of collocation points
    :param yi: y-coords of collocation points
    :param a_mat: influence coefficient matrix excluding the wake influence
    :param x_wake: x-pos of trailing edge wake
    :param y_wake: y-pos of trailing edge wake
    :param ni: array containing the normal vectors
    :param wake_gamma: circulation of the wake vortex (unit strength)

    :return: new influence matrix (a_mat) and RHS contribution of the wake vortices (v_norm)
    """

    print('   ...Computing influence matrix.')
    # Create induced velocities array
    v_norm = np.zeros(Npan + 1)

    print('   ...Computing induced velocities.')

    # determine influence coefficient from latest wake vorticity
    for i in range(Npan):
        uv = lumpvor2d(xi[i], yi[i], x_wake[-1], y_wake[-1])
        a_mat[i, -1] = uv @ ni[:, i]

        # determine RHS contribution of wake vortices (except trailing edge wake vortex)
        for j, gamma in enumerate(wake_gamma):
            uv = lumpvor2d(xi[i], yi[i], x_wake[j+1], y_wake[j+1], gamma)  # size(x_wake) > size(wake_gamma)
            v_norm[i] = v_norm[i] + uv @ ni[:, i]

    return a_mat, v_norm


def compute_pressure_and_loads(u_inf, v_inf, dc, dt, gamma_vec, gamma_vec_old, theta, gamma_steady, rho=1.225):
    """
    Computes the lift and drag based on the velocity components and circulation

    :param rhs: current RHS vector of linear system
    :param gamma_arr: 2D array containing the previous steps
    :param tan_vec: 2D array containing the tangential unit vector of the airfoil panels
    :param l_panels: 1D array containing the panel length
    :param dt: timestep

    :return: pressure difference, lift, moment, drag
    """

    # freestream speed
    v_eff = np.sqrt(u_inf ** 2 + v_inf ** 2)

    # UNSTEADY LIFT COEFFICIENT
    # compute time derivative contribution
    gamma_k_old = np.cumsum(gamma_vec_old)
    gamma_k = np.cumsum(gamma_vec)

    # pressure difference
    delta_p = rho * (v_eff * gamma_vec / dc) + (gamma_k - gamma_k_old) / dt

    # lift coefficient per panel and airfoil lift coefficient
    cl_per_sec = delta_p / (0.5 * rho * v_eff**2)
    cl = np.sum(cl_per_sec * dc * np.cos(theta))

    # STEADY LIFT COEFFICIENT
    cl_per_sec_ss = 2 * gamma_steady / v_eff / dc
    cl_ss = np.sum(cl_per_sec_ss * dc)  # weighted average of the panel lengths

    return delta_p, cl, cl_ss


@nb.njit
def roll_vortex_wake(x_vor, y_vor, gamma_airfoil, x_wake, y_wake, gamma_wake, u_inf, dt):
    """
    Computes new location of vortex wake as it is convected by the local stream velocity
    :param x_vor: x-coords of vortices on airfoil
    :param y_vor: y-coords of vortices on airfoil
    :param x_wake: x-coords of vortices in the wake
    :param y_wake: y-coords of vortices in the wake
    :return: new x_wake and y_wake
    """

    uw_mat = np.zeros((len(x_wake), 2))

    for i, (x_wake_i, y_wake_i) in enumerate(zip(x_wake, y_wake)):
        uw = np.zeros(2)  # initialize variable to store total induced velocity
        # contribution from the vortices of the airfoil
        for x_vor_j, y_vor_j, gamma_airfoil_j in zip(x_vor, y_vor, gamma_airfoil):
            uw += lumpvor2d(x_wake_i, y_wake_i, x_vor_j, y_vor_j, gamma_airfoil_j)
        # contribution from the vortices in the wake
        for iv, (x_wake_k, y_wake_k, gamma_wake_k) in enumerate(zip(x_wake, y_wake, gamma_wake)):
            if iv != i:  # ignore self-induction (div. by zero)
                uw += lumpvor2d(x_wake_i, y_wake_i, x_wake_k, y_wake_k, gamma_wake_k)
        uw_mat[i] = uw

    # determine change in (x,y) coordinates of wake vortices
    x_wake = x_wake + uw_mat[:, 0] * dt
    y_wake = y_wake + uw_mat[:, 1] * dt

    return  x_wake, y_wake


def steady_VP(y, x, Npan, Npan_flap, alpha, a_flap, c, c_flap, U_0, rho, key):

    print('   ...Creating geometry.')

    # Add a flap (OPTIONAL)
    if enable_flap:

        x = x * (1 - c_flap)

        Nf_points = Npan_flap + 1
        x_flap = np.arange(1, Nf_points + 1)
        x_flap = c_flap * c/2 * (1 - np.cos((x_flap - 1) * np.pi/(Nf_points - 1)))
        y_flap = np.copy(x_flap) * 0.

        xp_flap = x_flap * np.cos(a_flap) + y_flap * np.sin(a_flap)
        yp_flap = -x_flap * np.sin(a_flap) + y_flap * np.cos(a_flap)

        x = np.concatenate((x, xp_flap[1:]+ (1 - c_flap)))
        y = np.concatenate((y, yp_flap[1:]))

        Npan = Npan + Npan_flap

    # center LE at the origin
    x = x - c/4

    # Rotate airfoil clockwise
    xp = x * np.cos(alpha) + y * np.sin(alpha)
    yp = -x * np.sin(alpha) + y * np.cos(alpha)

    if key == 1:

        key = 0
        return xp, yp

    # Calculate dx,dy,dc component per panel (dc = panel length)
    dx = np.delete(np.roll(xp, -1) - xp, -1)
    dy = np.delete(np.roll(yp, -1) - yp, -1)
    dc = np.sqrt(dx**2 + dy**2)

    # Further induced geometry calculations
    alpha_i = np.arctan2(dy, dx)                            # Induced AoA by panel slope
    ni = np.array([np.sin(-alpha_i), np.cos(-alpha_i)])     # Normal vector; First index = x, second index = y

    # Calculate x,y coordinates of the quarter cord (c4) and collocation points (cp)
    xc4 = xp[0:-1] + dx/4
    yc4 = yp[0:-1] + dy/4

    xcp = xp[0:-1] + dx * (3/4)
    ycp = yp[0:-1] + dy * (3/4)

    print('   ...Solving linear system for circulation strengths.')

    # Solve system
    aij_mat = aijmatrix(xcp, ycp, xc4, yc4, Npan, ni)   # Influence matrix
    RHS = U_0 * np.sin(alpha_i) * np.ones(Npan)         # RHS vector

    gammamatrix = np.linalg.solve(aij_mat, RHS)         # Find circulation of each vortex point

    print('   ...Calculating lift and pressure.\n')

    # Secondary computations
    dLj = rho * U_0 * gammamatrix           # Lift difference
    L = np.sum(dLj)                         # Total Lift
    Cl = L / (0.5 * rho * U_0 ** 2 * c)     # Lift coefficient

    dpj = rho * U_0 * gammamatrix / dc      # Pressure difference
    dcpj = dpj/(0.5 * rho * (U_0**2))       # Pressure coefficient difference between upper and lower surface

    return xc4, yc4, dcpj, Cl, gammamatrix, xp, yp


def unsteady_VP(y, x, Npan, Npan_flap, alpha_arr, dalpha_arr, a_flap, c, c_flap, U_0, rho):

    print('   ...Initialize influence matrix.')

    # Add a flap (OPTIONAL)
    if enable_flap:

        x = x * (1 - c_flap)

        Nf_points = Npan_flap + 1
        x_flap = np.arange(1, Nf_points + 1)
        x_flap = c_flap * c / 2 * (1 - np.cos((x_flap - 1) * np.pi / (Nf_points - 1)))
        y_flap = np.copy(x_flap) * 0.

        xp_flap = x_flap * np.cos(a_flap) + y_flap * np.sin(a_flap)
        yp_flap = -x_flap * np.sin(a_flap) + y_flap * np.cos(a_flap)

        x = np.concatenate((x, xp_flap[1:] + (1 - c_flap)))
        y = np.concatenate((y, yp_flap[1:]))

        Npan = Npan + Npan_flap

    # center LE at the origin
    x = x - c / 4

    # Calculate dx,dy,dc component per panel (dc = panel length)
    dx = np.delete(np.roll(x, -1) - x, -1)
    dy = np.delete(np.roll(y, -1) - y, -1)

    # Further induced geometry calculations
    alpha_i = np.arctan2(dy, dx)                            # Induced AoA by panel slope
    ni = np.array([np.sin(-alpha_i), np.cos(-alpha_i)])     # Normal vector; First index = x, second index = y

    # Calculate x,y coordinates of the quarter cord (c4) and collocation points (cp)
    xc4 = x[0:-1] + dx / 4
    yc4 = y[0:-1] + dy / 4

    xcp = x[0:-1] + dx * (3 / 4)
    ycp = y[0:-1] + dy * (3 / 4)

    # Solve system
    aij_mat = aijmatrix(xcp, ycp, xc4, yc4, Npan, ni)  # influence matrix
    # add additional column THEN add new row to account for shed vortex
    aij_mat = np.vstack((np.hstack((aij_mat, np.zeros((Npan, 1)))), np.ones((1, Npan + 1))))

    # Initial variables
    gamma_vec = np.array([])
    fix_gamma = np.zeros((len(trange), 1))
    wake_gamma = np.array([])
    gamma_arr = np.zeros((len(trange) + 1, Npan))
    xwake = np.array([x[-1]])
    ywake = np.array([0])
    cl_unsteady_arr = np.zeros(len(trange))
    cl_steady_arr = np.zeros(len(trange))
    delta_p_arr = np.zeros((len(trange), Npan))

    # Storage arrays for all time steps
    xp_arr = np.zeros((Npan + 1, len(trange)))
    yp_arr = np.zeros((Npan + 1, len(trange)))

    print('   ...Start time loop.\n')

    for t in range(len(trange)):  # not actually time but nth timestep

        print(f"   Time step: {t+1}/{len(trange)}")
        print('   ...Creating geometry.')

        if enable_gust and trange[t] > gust_start:

            print('   ...Gust active')
            V_0 = v_gust     # (m/s) free-stream velocity in y

        else:

            V_0 = 0     # (m/s) free-stream velocity in y

        galpha = np.arctan2(V_0, U_0)

        xp = x * np.cos(alpha_arr[t]) + y * np.sin(alpha_arr[t])
        yp = -x * np.sin(alpha_arr[t]) + y * np.cos(alpha_arr[t])

        xp_arr[:, t] = xp
        yp_arr[:, t] = yp

        # Calculate dx, dy, dc component per panel (dc = panel length)
        dx = np.delete(np.roll(xp, -1) - xp, -1)
        dy = np.delete(np.roll(yp, -1) - yp, -1)
        dc = np.sqrt(dx**2 + dy**2)

        # Further induced geometry calculations
        alpha_i = np.arctan2(dy, dx)                            # Induced AoA by panel slope
        ni = np.array([np.sin(-alpha_i), np.cos(-alpha_i)])     # Normal vector; First index = x, second index = y
        ti = np.array([np.cos(alpha_i), -np.sin(alpha_i)])      # Tan. vector; first index = x, second index = y

        # Calculate x,y coordinates of the quarter cord (c4) and collocation points (cp)
        xc4 = xp[0:-1] + dx/4
        yc4 = yp[0:-1] + dy/4

        xcp = xp[0:-1] + dx * (3/4)
        ycp = yp[0:-1] + dy * (3/4)

        print('   ...Solving linear system for circulation strengths.')

        # Sum circulations except last one for RHS element of Kelvin condition (last row)
        f_gamma = np.sum(gamma_vec[:-1])
        fix_gamma[t] = f_gamma

        # Calculate last column influence matrix (influence of trailing edge vortex wake)
        aij_mat, v_norm = aijmatrix2(aij_mat, xcp, ycp, xwake, ywake, ni, wake_gamma)

        # Solve steady-state system
        RHS_ss = U_0 * np.sin(alpha_arr[t]) * np.ones(Npan)
        gamma_ss_vec = -(np.linalg.inv(aij_mat[:-1, :-1]) @ RHS_ss)

        # Pitching RHS | eqn 13.116
        xcp = xcp.reshape([Npan, 1])
        ycp = ycp.reshape([Npan, 1])
        a = np.concatenate((xcp, ycp, np.zeros((Npan, 1))), axis=1)
        b = np.concatenate((np.zeros((Npan, 2)), np.ones((Npan, 1)) * dalpha_arr[t]), axis=1)
        v_pitch = np.cross(-a, b)
        v_pitch_n = np.concatenate(((v_pitch[:, :2]*np.asarray(ni).T).sum(axis=1), [0]), axis=0)
        RHS = U_0 * np.sin(alpha_arr[t]) * np.ones(Npan + 1) + V_0 * np.cos(alpha_arr[t]) * np.ones(Npan + 1) + v_norm + v_pitch_n
        RHS[-1] = -f_gamma

        # directly solve system and obtain new circulation over airfoil and trailing edge vortex wake
        gamma_vec = -(np.linalg.inv(aij_mat) @ RHS)
        wake_gamma = np.append(wake_gamma, gamma_vec[-1])   # log circulation of wake
        gamma_arr[t + 1] = gamma_vec[:-1]                   # log circulation of each panel

        print('   ...Calculating lift and pressure.')
        # Secondary computations
        delta_p, cl, cl_ss = compute_pressure_and_loads(U_0, V_0, dc, dt, gamma_arr[t+1], gamma_arr[t], alpha_arr[t],
                                                        gamma_ss_vec, rho)

        # compute wake sheet roll-up
        # print('   ...Computing wake sheet roll-up.')
        # xwake, ywake = roll_vortex_wake(xc4, yc4, gamma_vec[:-1], xwake, ywake, wake_gamma, dt, U_0)

        # add new vortex wake
        xwake = xwake + U_0 * dt
        xwake_new = xp[-1] + 0.25 * (xwake[t] - xp[-1])
        ywake_new = yp[-1] + 0.25 * (ywake[t] - yp[-1])
        xwake = np.append(xwake, xwake_new)
        ywake = np.append(ywake, ywake_new)

        # log pressure and loads
        cl_unsteady_arr[t] = cl
        cl_steady_arr[t] = cl_ss
        delta_p_arr[t] = delta_p

    # sort in dictionary
    results_unsteady = {"delta_p": delta_p_arr, "cl_unsteady": cl_unsteady_arr, "cl_steady": cl_steady_arr}

    return xc4, yc4, xp, yp, gamma_vec, wake_gamma, xwake, ywake, results_unsteady, xp_arr, yp_arr

# ---------------------------------- #
# Solver
# ---------------------------------- #

# Number of airfoil nodes
Npoints = Npan + 1

if enable_cos_dist:

    # Discretization for X using cosine distribution
    x  = np.arange(1, Npoints + 1)
    x = c/2 * (1 - np.cos((x - 1) * np.pi/(Npoints - 1)))

else:

    x = np.linspace(0, c, Npoints)

if apply_camber:

    # naca 4 series camber
    y = naca4(c, camber, 0.4, x)

else:

    # Flat plate
    y = np.copy(x) * 0.

if enable_pitching:

    alpha_arr = amp * np.sin(omega * trange)            # AoA log
    dalpha_arr = amp * omega * np.cos(omega * trange)   # Derivative of AoA log

if plot_ss_cl_curve:

    print('...Building lift curve.')
    print('...Running solver.\n')

    cl_arr = np.zeros(len(alpha_range))  # Lift coefficient log

    for i, alpha in enumerate(alpha_range):

        print(f"   Alpha: {alpha} deg")
        temp = steady_VP(y, x, Npan, Npan_flap, np.deg2rad(alpha), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=0)
        cl_arr[i] = temp[3]

if plot_velocity_field or plot_pressure_field:

    print('...Running solver.\n')

    # velocity and pressure logs
    u = np.ones((len(X[:, 0]), len(Y[0, :]))) * U_0
    v = np.zeros((len(X[:, 0]), len(Y[0, :])))
    v_map = np.zeros((len(X[:, 0]), len(Y[0, :])))
    cp_map = np.zeros((len(X[:, 0]), len(Y[0, :])))

    if enable_pitching:

        result = unsteady_VP(y, x, Npan, Npan_flap, alpha_arr, dalpha_arr, np.deg2rad(a_flap), c, c_flap, U_0, rho)

        xx = result[0]
        yy = result[1]
        xp = result[2]
        yp = result[3]
        gammaB = result[4]
        gammaW = result[5]
        xwake = result[6]
        ywake = result[7]
        xp_arr = result[9]
        yp_arr = result[10]

        v_map, cp_map = compute_velocity_field_us(U_0, X, Y, xp, yp, gammaB, gammaW, xwake, ywake)

    else:

        result = steady_VP(y, x, Npan, Npan_flap, np.deg2rad(AoA), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=0)

        xx = result[0]
        yy = result[1]
        gammaM = result[4]
        xp = result[5]
        yp = result[6]

        print('...Creating velocity and pressure distribution.\n')

        # Build velocity and pressure distribution over background mesh
        v_map, cp_map = compute_velocity_field_ss(U_0, X, Y, xx, yy, gammaM)
    print('')

# ---------------------------------- #
# Figures
# ---------------------------------- #

print('...Plotting figures.')
plt.close('all')

if plot_backmesh:

    temp = steady_VP(y, x, Npan, Npan_flap, np.deg2rad(AoA), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=1)
    xp = temp[0]
    yp = temp[1]
    plt.figure("Background Mesh")
    plt.title(r'Local Refinement')
    plt.scatter(X, Y, s=0.5)
    plt.plot(xp, yp, '-k', lw=2.)
    plt.xlabel('x/c [-]')
    plt.ylabel('y/c [-]')

if plot_ss_cl_curve:

    plt.figure("CL")
    plt.title(r'$C_l$-$\alpha$ curve')
    plt.plot(alpha_range, cl_arr, label='Panel Method', lw=1.2)
    plt.plot(alpha_range, 2*np.pi*np.deg2rad(alpha_range), '--', label='Analytic', lw=1.2)
    plt.xlabel(r'Angle of attack $\alpha$ (deg)')
    plt.ylabel(r'Lift coefficient $C_l$ (-)')
    plt.grid('True')
    plt.legend()

if plot_velocity_field:

    if enable_pitching:

        levels = 400
        men = np.mean(v_map)
        rms = np.sqrt(np.mean((v_map-men) ** 2))
        vmin = round(men - 3 * rms, 1)
        vmax = round(men + 3 * rms, 1)
        # vmin = round(np.amin(v_map))
        # vmax = round(np.amax(v_map))
        level_boundaries = np.linspace(vmin, vmax, levels + 1)

    else:

        levels = 400
        men = np.mean(v_map)
        rms = np.sqrt(np.mean((v_map) ** 2))
        vmin = round(men - 0.5 * rms)
        vmax = round(men + 0.5 * rms)
        # vmin = round(np.amin(v_map))
        # vmax = round(np.amax(v_map))
        level_boundaries = np.linspace(vmin, vmax+0.001, levels + 1)

    v_map[v_map > vmax] = vmax * 0.999
    v_map[v_map < vmin] = vmin * 1.001
    plt.figure('Velocity Magnitude', dpi=100, constrained_layout=True)
    cmap = plt.get_cmap('jet')
    plt.title('Velocity Magnitude', fontsize=16)
    cf = plt.contourf(X, Y, v_map, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
    clb = plt.colorbar(
        ScalarMappable(norm=cf.norm, cmap=cf.cmap),
        ticks=np.round(np.linspace(vmin, vmax, 4), 1),
        boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2,
        )
    clb.ax.set_title(r'$V$ (m/s)', fontsize=14)
    plt.plot(xp_arr[:, 2], yp_arr[:, 2], '-k', lw=2.)
    plt.xlabel('x/c [-]', fontsize=16)
    plt.ylabel('y/c [-]', fontsize=16)

if plot_pressure_field:

    if enable_pitching:

        levels = 400
        # men = np.mean(cp_map)
        # rms = np.sqrt(np.mean((cp_map-men) ** 2))
        # pmin = round(men - 3 * rms)
        # pmax = round(men + 3 * rms)
        pmin = -0.5
        pmax = 0.5
        # pmin = round(np.amin(cp_map))
        # pmax = round(np.amax(cp_map))
        level_boundaries = np.linspace(pmin, pmax, levels + 1)

    else:

        levels = 400
        # men = np.mean(cp_map)
        # rms = np.sqrt(np.mean((cp_map-men) ** 2))
        # pmin = round(men - 0.5 * rms)
        # pmax = round(men + 0.5 * rms)
        pmin = -1
        pmax = 1
        # pmin = round(np.amin(cp_map))
        # pmax = round(np.amax(cp_map))
        level_boundaries = np.linspace(pmin, pmax, levels + 1)

    cp_map[cp_map > pmax] = pmax
    cp_map[cp_map < pmin] = pmin
    plt.figure('Pressure Distribution')
    cmap = plt.get_cmap('jet')
    plt.title('Pressure Distribution')
    cf2 = plt.contourf(X, Y, cp_map, levels=levels, vmin=pmin, vmax=pmax, cmap=cmap)
    clb = plt.colorbar(
        ScalarMappable(norm=cf2.norm, cmap=cf2.cmap),
        ticks=np.arange(pmin, pmax + 0.2, 0.2),
        boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, )
    clb.ax.set_title(r'$C_p$ (-)')
    plt.plot(xp, yp, '-k', lw=2.)
    plt.xlabel('x/c [-]')
    plt.ylabel('y/c [-]')

if plot_camber:

    # Calculate Y-coordinates according to naca4 airfoil
    ycb = naca4(c, camber, 0.4, x)      # Calculate camber line
    yflat = np.copy(x) * 0.             # Flat plate

    plt.figure("Camber Lines", dpi=100, constrained_layout=True)
    plt.title('Camber lines', fontsize=16)
    plt.plot(x, yflat, '-ob', markevery=5, label='Flat plate')
    plt.plot(x, ycb, '-or', markevery=5, label=str(camber*100)+'% Camber')
    plt.xlabel('x/c [-]', fontsize=16)
    plt.ylabel('y/c [-]', fontsize=16)
    plt.xticks(np.arange(min(x), max(x) + 0.1, 0.1))
    plt.yticks(np.arange(-0.4, 0.6, 0.1))
    plt.grid('True')
    plt.legend(prop={"size": 14})

if plot_deltaP_comp:

    # Calculate Y-coordinates according to naca4 airfoil
    ycb = naca4(c, camber, 0.4, x)   # Calculate camber line
    yflat = np.copy(x) * 0.          # Flat plate

    # Results needed for plotting
    result_flat = steady_VP(yflat, x, Npan, Npan_flap, np.deg2rad(AoA), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=0)
    result_c4 = steady_VP(ycb, x, Npan, Npan_flap, np.deg2rad(AoA), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=0)

    plt.figure("Pressure Difference")
    plt.title("Pressure Difference")
    plt.plot(result_flat[0]/np.cos(np.deg2rad(AoA)) + c/4, result_flat[2], '-b', label='Flat Plate')
    plt.plot(result_c4[0]/np.cos(np.deg2rad(AoA)) + c/4, result_c4[2], '-r', label=str(camber*100)+'% Camber')
    plt.ylim(-3 , 6)
    plt.xlabel('x/c [-]')
    plt.ylabel(r'$Cp_l - Cp_u$ [-]')
    plt.grid('True')
    plt.legend()

if (plot_CLcirc and enable_pitching):
    plot_circulatory_loads(alpha_arr, dalpha_arr, result[8]["cl_unsteady"], result[8]["cl_steady"], xwake, ywake, U_0,
                           c, trange, plot_wake=True)

if plot_dt_comp:

    colors = ["r", "b", "g", "m"]
    style = ["-", "--", "-.", "."]

    #dt_list = [0.5, 0.1, 0.05, 0.01]
    dt_list = [1.0, 0.5, 0.1, 0.05]

    for i, dt in enumerate(dt_list):

        trange = np.arange(start, stop + dt, dt)  # Time log
        alpha_arr = amp * np.sin(omega * trange)  # AoA log
        dalpha_arr = amp * omega * np.cos(omega * trange)  # Derivative of AoA log

        temp = unsteady_VP(y, x, Npan, Npan_flap, alpha_arr, dalpha_arr, np.deg2rad(a_flap), c, c_flap, U_0, rho)
        result = temp[8]
        values = result.values()
        values_list = list(values)
        cl_us = values_list[1]
        cl_ss = values_list[2]

        plt.figure("Time step convergence")
        plt.title("Time step convergence")
        plt.plot(trange, cl_ss, style[i]+colors[i], label='dt ='+str(dt))
        #plt.ylim(-3, 6)
        plt.xlabel('t [s]')
        plt.ylabel(r'$C_l [-]$')
        plt.grid('True')
        plt.legend()

plt.show()
print('\nDone.')