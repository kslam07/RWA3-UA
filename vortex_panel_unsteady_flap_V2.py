
# THIN AIRFOIL VORTEX PANEL METHOD #

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numba as nb  # trying out write unvectorized code with Numba jitting
import warnings
warnings.simplefilter('ignore', category=nb.errors.NumbaPerformanceWarning)

# ---------------------------------- #
# Flags
# ---------------------------------- #

print('Start.\n')
print('...Setting flags based on user input.')

enable_flap         = False
enable_pitching     = True
add_meshrefinement  = True
visualize_wake      = False

plot_backmesh       = False
plot_camber         = False
plot_cl_curve       = False
plot_velocity_field = True
plot_pressure_field = False
plot_deltaP         = False

# ---------------------------------- #
# Geometry and operations parameters
# ---------------------------------- #

print('...Read global parameters.')

# Airfoil
Npan = 40               # Number of panels
c = 1                   # (m) chord length

# Flap
c_flap = 0.3            # Fraction of chord length
a_flap = 15             # (deg) Flap deflection in degrees
Npan_flap = 10          # Number of panels on flap

# Mesh
xres = 50               # Grid discretization in x direction
yres = 50               # Grid discretization in y direction

# Operations
rho = 1.225             # (kg/m^3) free-stream density
U_0 = 10                # (m/s) free-stream velocity
k = 0.1                 # (Hz) Reduced frequency: 0.02, 0.05, 0.1
omega = k*2*U_0/c       # (Hz) Frequency of the unsteadiness
amp = 15                # (deg) Amplitude of the pitching motion

# Time
start = 0                                   # Start time
stop = np.pi/(8*omega)                      # Stop time
dt = 0.002                                  # Time step
trange = np.arange(start, stop + dt, dt)    # Time log

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
    yrange = np.linspace(-1.5 * c, 1.5 * c, yres)*(1 - 0.9 * np.sin(np.linspace(0, np.pi, yres)))

else:

    xrange = np.linspace(-1.0 * c, wl * c, xres)
    yrange = np.linspace(-1.5 * c, 1.5 * c, yres)

# Create grid
[X, Y] = np.meshgrid(xrange, yrange)

# ---------------------------------- #
# Vortex Panel Functions
# ---------------------------------- #


def naca4(c, m, p, x):  # Calculate corresponding Y components of camber line for naca 4series

    # c = hord length,m = max camber, p = max camber postion, x are x coordinates

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

    if r_vortex_sq < 1e-8:  # some arbitrary threshold
        return np.array([0.0, 0.0])

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
    :param x_twake: x-pos of trailing edge wake
    :param y_twake: y-pos of trailing edge wake
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
        for j in range(len(wake_gamma[:-1])):
            uv = lumpvor2d(xi[i], yi[i], x_wake[j], y_wake[j])
            v_norm[i] = v_norm[i] + uv @ ni[:, i]

    return a_mat, v_norm

@nb.njit
def roll_vortex_wake(x_vor, y_vor, gamma_airfoil, x_wake, y_wake, gamma_wake, dt):
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
        for x_wake_k, y_wake_k, gamma_wake_k in zip(x_wake, y_wake, gamma_wake):
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
    ni = np.array([np.sin(-alpha_i), np.cos(-alpha_i)])    # Normal vector; First index = x, second index = y

    # Calculate x,y coordinates of the quarter cord (c4) and collocation points (cp)
    xc4 = xp[0:-1] + dx/4
    yc4 = yp[0:-1] + dy/4

    xcp = xp[0:-1] + dx * (3/4)
    ycp = yp[0:-1] + dy * (3/4)

    print('   ...Solving linear system for circulation strengths.')

    # Solve system #
    aij_mat = aijmatrix(xcp, ycp, xc4, yc4, Npan, ni)   # aij matrix
    RHS = U_0 * np.sin(alpha_i) * np.ones(Npan)  # RHS vector
    # if enable_flap:
    #
    #     RHS = U_0 * np.sin(alpha_i + a_flap) * np.ones(Npan)         # RHS vector
    #
    # else:
    #
    #     RHS = U_0 * np.sin(alpha_i) * np.ones(Npan)         # RHS vector

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
    dc = np.sqrt(dx ** 2 + dy ** 2)

    # Further induced geometry calculations
    alpha_i = np.arctan2(dy, dx)  # Induced AoA by panel slope
    ni = np.array([np.sin(-alpha_i), np.cos(-alpha_i)])  # Normal vector; First index = x, second index = y

    # Calculate x,y coordinates of the quarter cord (c4) and collocation points (cp)
    xc4 = x[0:-1] + dx / 4
    yc4 = y[0:-1] + dy / 4

    xcp = x[0:-1] + dx * (3 / 4)
    ycp = y[0:-1] + dy * (3 / 4)

    # Solve system #
    aij_mat = aijmatrix(xcp, ycp, xc4, yc4, Npan, ni)  # aij matrix
    # add additional column THEN add new row
    aij_mat = np.vstack((np.hstack((aij_mat, np.zeros((Npan, 1)))), np.ones((1, Npan + 1))))

    # Initial variables
    gammamatrix = np.zeros(1)
    fix_gamma = np.zeros((1, len(trange)))
    wake_gamma = np.zeros(1)
    xwake = np.array([x[-1] - c/4])
    ywake = np.array([0])

    # Storage arrays for all time steps
    xp_arr = np.zeros((Npan + 1, len(trange)))
    yp_arr = np.zeros((Npan + 1, len(trange)))
    v_map_arr = np.zeros((len(X[:, 1]), len(Y[1, :]), len(trange)))
    cp_map_arr = np.zeros((len(X[:, 1]), len(Y[1, :]), len(trange)))
    gamma_arr = np.zeros((Npan + 1, len(trange)))

    print('   ...Start time loop.')

    for t in range(len(trange)):

        print(f"   Time step: {t+1}/{len(trange)}")
        print('   ...Creating geometry.')

        # center LE at the origin
        x = x - c/4

        # Rotate airfoil clockwise
        xp = x * np.cos(alpha_arr[t]) + y * np.sin(alpha_arr[t])
        yp = -x * np.sin(alpha_arr[t]) + y * np.cos(alpha_arr[t])

        xp_arr[:, t] = xp
        yp_arr[:, t] = yp

        # Calculate dx,dy,dc component per panel (dc = panel length)
        dx = np.delete(np.roll(xp, -1) - xp, -1)
        dy = np.delete(np.roll(yp, -1) - yp, -1)
        dc = np.sqrt(dx**2 + dy**2)

        # Further induced geometry calculations
        alpha_i = np.arctan2(dy, dx)                            # Induced AoA by panel slope
        ni = np.array([np.sin(-alpha_i), np.cos(-alpha_i)])    # Normal vector; First index = x, second index = y

        # Calculate x,y coordinates of the quarter cord (c4) and collocation points (cp)
        xc4 = xp[0:-1] + dx/4
        yc4 = yp[0:-1] + dy/4

        xcp = xp[0:-1] + dx * (3/4)
        ycp = yp[0:-1] + dy * (3/4)

        print('   ...Solving linear system for circulation strengths.')

        # Sum circulations except last one
        f_gamma = np.sum(gammamatrix[:-1])
        fix_gamma[t] = f_gamma

        # Calculate influence matrix related to last shedded vortex
        # Calculate wake induced velocities
        aij_mat, v_norm = aijmatrix2(aij_mat, xcp, ycp, xwake, ywake, ni, wake_gamma)   # aij matrix

        # Non-circulatory
        RHS = U_0 * np.sin(alpha_arr[t]) * np.ones(Npan+1)

        # Pitching RHS
        xcp = xcp.reshape([Npan, 1])
        ycp = ycp.reshape([Npan, 1])
        a = np.concatenate((xcp, ycp, np.zeros((Npan, 1))), axis=1)
        b = np.concatenate((np.zeros((Npan, 2)), np.ones((Npan, 1)) * dalpha_arr[t]), axis=1)
        v_pitch = np.cross(-a, b)
        v_pitch_n = np.concatenate(((v_pitch[:, :2]*np.asarray(ni).T).sum(axis=1), [0]), axis=0)
        RHS = U_0 * np.sin(alpha_arr[t]) * np.ones(Npan + 1) + v_norm + v_pitch_n
        RHS[-1] = -f_gamma

        #gammamatrix = np.linalg.solve(aij_mat, RHS)         # Find circulation of each vortex point
        gamma_vec = np.linalg.inv(aij_mat) @ RHS
        # wake_gamma = np.append(wake_gamma, gammamatrix[-1])
        # print(gammamatrix[:-1].shape)
        # gamma_arr[t] = np.sum(gammamatrix[:-1])

        print('   ...Calculating lift and pressure.\n')

        # Secondary computations
        # todo: compute velocity components, pressures, and loads
        # dLj = rho * U_0 * gamma_vec           # Lift difference
        # L = np.sum(dLj)                         # Total Lift
        # Cl = L / (0.5 * rho * U_0 ** 2 * c)     # Lift coefficient
        #
        # dpj = rho * U_0 * gamma_vec / dc      # Pressure difference
        # dcpj = dpj/(0.5 * rho * (U_0**2))       # Pressure coefficient difference between upper and lower surface

        # compute wake sheet roll-up
        xwake, ywake = roll_vortex_wake(xc4, yc4, gamma_vec, xwake, ywake, wake_gamma, dt)

    return xc4, yc4, dcpj, Cl, gamma_vec, xp, yp

# ---------------------------------- #
# Solver
# ---------------------------------- #

# Discretization for X using cosine distribution
Npoints = Npan + 1
x  = np.arange(1, Npoints + 1)
x = c/2 * (1 - np.cos((x - 1) * np.pi/(Npoints - 1)))

# Flat plate
y = np.copy(x) * 0.

if enable_pitching:

    alpha_arr = amp * np.sin(omega * trange)            # AoA log
    dalpha_arr = amp * omega * np.cos(omega * trange)   # Derivative of AoA log

if plot_cl_curve:

    print('...Building lift curve.')
    print('...Running solver.\n')

    alpha_range = np.arange(-4, 15)         # Range of AoA
    cl_arr = np.zeros(len(alpha_range))     # Lift coefficient log

    for i, alpha in enumerate(alpha_range):

        print(f"   Alpha: {alpha} deg")
        temp = steady_VP(y, x, Npan, Npan_flap, np.deg2rad(alpha), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=0)
        cl_arr[i] = temp[3]

if plot_velocity_field or plot_pressure_field:

    print('...Running solver.\n')

    if enable_pitching:

        result = unsteady_VP(y, x, Npan, Npan_flap, alpha_arr, dalpha_arr, np.deg2rad(a_flap), c, c_flap, U_0, rho)

    else:

        result = steady_VP(y, x, Npan, Npan_flap, np.deg2rad(15), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=0)

    xx = result[0]
    yy = result[1]
    gammaM = result[4]
    xp = result[5]
    yp = result[6]

    # Build velocity and pressure distribution
    u = np.ones((len(X), len(Y))) * U_0
    v = np.zeros((len(X), len(Y)))
    v_map = np.zeros((len(X), len(Y)))
    cp_map = np.zeros((len(X), len(Y)))

    print('...Creating velocity and pressure distribution.\n')

    for i in range(len(X[:, 1])):

        print('   ...Row:', i+1)

        for j in range(len(Y[1, :])):

            for g, gamma in enumerate(gammaM):

                uv = indvel(gamma, X[i, j], Y[i, j], xx[g], yy[g])
                u[i, j] = u[i, j] + uv[0]
                v[i, j] = v[i, j] + uv[1]

            v_map[i, j] = np.sqrt(u[i, j]**2 + v[i, j]**2)
            cp_map[i, j] = 1 - (v_map[i, j]/U_0)**2

    print('')

# ---------------------------------- #
# Figures
# ---------------------------------- #

print('...Plotting figures.')
plt.close('all')

if plot_backmesh:

    temp = steady_VP(y, x, Npan, Npan_flap, np.deg2rad(8), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=1)
    xp = temp[0]
    yp = temp[1]
    plt.figure("Background Mesh")
    plt.title(r'Local Refinement')
    plt.scatter(X, Y, s=0.5)
    plt.plot(xp, yp, '-k', lw=2.)
    plt.xlabel('x/c [-]')
    plt.ylabel('y/c [-]')

if plot_cl_curve:

    plt.figure("CL")
    plt.title(r'$C_l$-$\alpha$ curve')
    plt.plot(alpha_range, cl_arr, label='Panel Method', lw=1.2)
    plt.plot(alpha_range, 2*np.pi*np.deg2rad(alpha_range), '--', label='Analytic', lw=1.2)
    plt.xlabel(r'Angle of attack $\alpha$ (deg)')
    plt.ylabel(r'Lift coefficient $C_l$ (-)')
    plt.grid('True')
    plt.legend()

if plot_velocity_field:

    levels = 400
    men = np.mean(v_map)
    rms = np.sqrt(np.mean(v_map ** 2))
    vmin = round(men - 0.5 * rms)
    vmax = round(men + 0.5 * rms)
    # vmin = round(np.amin(v_map))
    # vmax = round(np.amax(v_map))
    level_boundaries = np.linspace(vmin, vmax, levels + 1)

    plt.figure('Velocity Magnitude')
    cmap = plt.get_cmap('jet')
    plt.title('Velocity Magnitude')
    cf = plt.contourf(X, Y, v_map, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
    clb = plt.colorbar(
        ScalarMappable(norm=cf.norm, cmap=cf.cmap),
        ticks=range(vmin, vmax+2, 2),
        boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2,)
    clb.ax.set_title(r'$V$ (m/s)')
    plt.plot(xp, yp, '-k', lw=2.)
    plt.xlabel('x/c [-]')
    plt.ylabel('y/c [-]')

if plot_pressure_field:

    levels = 400
    vmin = -1
    vmax = 1
    # vmin = round(np.amin(cp_map))
    # vmax = round(np.amax(cp_map))
    level_boundaries = np.linspace(vmin, vmax, levels + 1)

    plt.figure('Pressure Distribution')
    cmap = plt.get_cmap('jet')
    plt.title('Pressure Distribution')
    cf2 = plt.contourf(X, Y, cp_map, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
    clb = plt.colorbar(
        ScalarMappable(norm=cf2.norm, cmap=cf2.cmap),
        ticks=range(vmin, vmax+1),
        boundaries=level_boundaries,
        values=(level_boundaries[:-1] + level_boundaries[1:]) / 2, )
    clb.ax.set_title(r'$C_p$ (-)')
    plt.plot(xp, yp, '-k', lw=2.)
    plt.xlabel('x/c [-]')
    plt.ylabel('y/c [-]')

if plot_camber:

    # Calculate Y-coordinates according to naca4 airfoil
    ycb4 = naca4(c, 0.04, 0.4, x)   # Calculate camber line for 4%
    yflat = np.copy(x) * 0.         # Flat plate

    # Results needed for plotting
    result_flat = steady_VP(yflat, x, Npan, Npan_flap, np.deg2rad(4), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=0)
    result_c4 = steady_VP(ycb4, x, Npan, Npan_flap, np.deg2rad(4), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=0)

    plt.figure("Camber Lines")
    plt.title('Camber lines')
    plt.plot(x, yflat, '-ob', markevery=5, label='Flat plate')
    plt.plot(x, ycb4, '-or', markevery=5, label='4% Camber')
    plt.xlabel('x/c [-]')
    plt.ylabel('y/c [-]')
    plt.xticks(np.arange(min(x), max(x) + 0.1, 0.1))
    plt.yticks(np.arange(-0.4, 0.6, 0.1))
    plt.grid('True')
    plt.legend()

if plot_deltaP:

    # Calculate Y-coordinates according to naca4 airfoil
    ycb4 = naca4(c, 0.04, 0.4, x)   # Calculate camber line for 4%
    yflat = np.copy(x) * 0.         # Flat plate

    # Results needed for plotting
    result_flat = steady_VP(yflat, x, Npan, Npan_flap, np.deg2rad(4), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=0)
    result_c4 = steady_VP(ycb4, x, Npan, Npan_flap, np.deg2rad(4), np.deg2rad(a_flap), c, c_flap, U_0, rho, key=0)

    plt.figure("Pressure Difference")
    plt.title("Pressure Difference")
    plt.plot(result_flat[0], result_flat[2], '-b', label='Flat Plate')
    plt.plot(result_c4[0], result_c4[2], '-r', label='4% Camber')
    plt.ylim(-3 , 6)
    plt.xlabel('x/c [-]')
    plt.ylabel(r'$Cp_l - Cp_u$ [-]')
    plt.grid('True')
    plt.legend()

plt.show()
print('\nDone.')
exit(0)