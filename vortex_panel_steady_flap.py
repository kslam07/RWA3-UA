
## VORTEX PANEL METHOD
from post_process import compute_velocity_field_ss

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numba as nb
import warnings

warnings.simplefilter('ignore', category=nb.errors.NumbaPerformanceWarning)

# ---------------------------------- #
# Flags
# ---------------------------------- #

enable_flap         = False

plot_backmesh       = False
plot_camber         = False
plot_cl_curve       = False
plot_velocity_field = True
plot_pressure_field = False
plot_deltaP         = False
plot_panel_density  = True

# ---------------------------------- #
# Geometry and operations parameters
# ---------------------------------- #

# Airfoil
Npan = 100             # Number of panels
res = 50               # Grid discretization in 1 direction
c = 1                  # (m) chord length

# Flap
c_flap = 0.3           # Fraction of chord length
a_flap = 15            # (deg) Flap deflection in degrees
Npan_flap = 10         # Number of panels on flap

# Operations
rho = 1.225                     # (kg/m^3) free-stream density
U_0 = 10                        # (m/s) free-stream velocity
arange = np.arange(-4, 15)      # Range of AoA
cl_arr = np.zeros(len(arange))  # Lift coefficient log

# ---------------------------------- #
# Create Grid
# ---------------------------------- #

print('...Creating background grid and refinement.')
# Add refinement in the vicinity of the flat plate
xrange = np.linspace(-1.0 * c, 2.0 * c, res)*(1 - 0.2 * np.sin(np.linspace(0, np.pi, res)))
yrange = np.linspace(-1.5 * c, 1.5 * c, res)*(1 - 0.9 * np.sin(np.linspace(0, np.pi, res)))
[X, Y] = np.meshgrid(xrange, yrange)

# ---------------------------------- #
# Vortex Panel Functions
# ---------------------------------- #

def naca4(c, m, p, x):  # Calculate corresponding Y components of camber line for naca 4series

    print('...Building NACA 4series camber line.')
    # c = hord length,m = max camber, p = max camber postion, x are x coordinates

    x1 = x[np.where(x <= p)]    # X components upto largest thickness point
    x2 = x[np.where(x > p)]     # X components after largest thickness point

    y1 = m / p ** 2 * (2 * p * x1 - x1 ** 2)                        # y with corresponding x1
    y2 = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x2 - x2 ** 2)    # y with corresponding x2
    y = np.concatenate((y1, y2))                                    # Combined y

    return y


def indvel(gammaj, x, y, xj, yj):  # Formula 11.1

    rj_sqrt = (x - xj) ** 2 + (y - yj) ** 2
    randommatrix = np.matrix('0 1; -1 0')
    xymatrix = np.matrix([[x - xj], [y - yj]])
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


@nb.njit
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


def calculation(y, x, Npan, Npan_flap, alpha, a_flap, c, c_flap, U_0, rho):

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

    # Calculate dx,dy,dc component per panel (dc = panel length)
    dx = np.delete(np.roll(xp, -1) - xp, -1)
    dy = np.delete(np.roll(yp, -1) - yp, -1)
    dc = np.sqrt(dx**2 + dy**2)

    # Further induced geometry calculations
    alpha_i = np.arctan2(dy, dx)                            # Induced AoA by panel slope
    ni = np.matrix([np.sin(-alpha_i), np.cos(-alpha_i)])    # Normal vector; First index = x, second index = y

    # Calculate x,y coordinates of the quarter cord (c4) and collocation points (cp)
    xc4 = xp[0:-1] + dx/4
    yc4 = yp[0:-1] + dy/4

    xcp = xp[0:-1] + dx * (3/4)
    ycp = yp[0:-1] + dy * (3/4)

    print('   ...Solving linear system for circulation strengths.')
    # Solve system #
    aij_mat = aijmatrix(xcp, ycp, xc4, yc4, Npan, ni)   # aij matrix
    RHS = U_0 * np.sin(alpha_i) * np.ones(Npan)         # RHS vector
    gammamatrix = np.linalg.solve(aij_mat, RHS)         # Find circulation of each vortex point

    print('   ...Calculating lift and pressure.\n')
    # Secondary computations
    dLj = rho * U_0 * gammamatrix           # Lift difference
    L = np.sum(dLj)                         # Total Lift
    Cl = L / (0.5 * rho * U_0 ** 2 * c)     # Lift coefficient

    dpj = rho * U_0 * gammamatrix / dc      # Pressure difference
    dcpj = dpj/(0.5 * rho * (U_0**2))       # Pressure coefficient difference between upper and lower surface

    return xc4, yc4, dcpj, Cl, gammamatrix, xp, yp, dLj, dc

# ---------------------------------- #
# Solver
# ---------------------------------- #

# Discretization for X using cosine distribution
Npoints = Npan + 1
x  = np.arange(1, Npoints + 1)
x = c/2 * (1 - np.cos((x - 1) * np.pi/(Npoints - 1)))

# Flat plate
y = np.copy(x) * 0.

if plot_cl_curve:

    print('...Building lift curve.')
    print('...Running solver.\n')
    for i, alpha in enumerate(arange):

        temp = calculation(y, x, Npan, Npan_flap, np.deg2rad(alpha), np.deg2rad(a_flap), c, c_flap, U_0, rho)
        cl_arr[i] = temp[3]

if plot_velocity_field or plot_pressure_field:

    print('...Running solver.\n')
    result = calculation(y, x, Npan, Npan_flap, np.deg2rad(15), np.deg2rad(a_flap), c, c_flap, U_0, rho)
    xx = result[0]
    yy = result[1]
    gammaM = result[4]
    xp = result[5]
    yp = result[6]

    print('...Creating velocity and pressure distribution.\n')
    v_map, cp_map = compute_velocity_field_ss(U_0, X, Y, xx, yy, gammaM)
    print('')

if plot_panel_density:

    print('...Building panel density plot.')
    print('...Running solver.\n')

    Nrange = [5, 20, 100]

    clx = np.array([])
    xxlist = np.array([])

    for i, Npan in enumerate(Nrange):

        # Discretization for X using cosine distribution
        Npoints = Npan + 1
        x = np.arange(1, Npoints + 1)
        x = c / 2 * (1 - np.cos((x - 1) * np.pi / (Npoints - 1)))

        # Flat plate
        y = np.copy(x) * 0.

        temp = calculation(y, x, Npan, Npan_flap, np.deg2rad(12), np.deg2rad(a_flap), c, c_flap, U_0, rho)
        clx = np.append(clx, temp[7] / (0.5 * rho * U_0 ** 2 * temp[8]))
        xxlist = np.append(xxlist, temp[0])

# ---------------------------------- #
# Figures
# ---------------------------------- #

print('...Plotting figures.')
plt.close('all')

if plot_backmesh:

    temp = calculation(y, x, Npan, Npan_flap, np.deg2rad(8), np.deg2rad(a_flap), c, c_flap, U_0, rho)
    xp = temp[5]
    yp = temp[6]
    plt.figure("Background Mesh")
    plt.title(r'Local Refinement')
    plt.scatter(X, Y, s=0.5)
    plt.plot(xp, yp, '-k', lw=2.)
    plt.xlabel('x/c [-]')
    plt.ylabel('y/c [-]')

if plot_cl_curve:

    plt.figure("CL")
    plt.title(r'$C_l$-$\alpha$ curve')
    plt.plot(arange, cl_arr, label='Panel Method', lw=1.2)
    plt.plot(arange, 2*np.pi*np.deg2rad(arange), '--', label='Analytic', lw=1.2)
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
    result_flat = calculation(yflat, x, Npan, Npan_flap, np.deg2rad(4), np.deg2rad(a_flap), c, c_flap, U_0, rho)
    result_c4 = calculation(ycb4, x, Npan, Npan_flap, np.deg2rad(4), np.deg2rad(a_flap), c, c_flap, U_0, rho)

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
    result_flat = calculation(yflat, x, Npan, Npan_flap, np.deg2rad(4), np.deg2rad(a_flap), c, c_flap, U_0, rho)
    result_c4 = calculation(ycb4, x, Npan, Npan_flap, np.deg2rad(4), np.deg2rad(a_flap), c, c_flap, U_0, rho)

    plt.figure("Pressure Difference")
    plt.title("Pressure Difference")
    plt.plot(result_flat[0], result_flat[2], '-b', label='Flat Plate')
    plt.plot(result_c4[0], result_c4[2], '-r', label='4% Camber')
    plt.ylim(-3 , 6)
    plt.xlabel('x/c [-]')
    plt.ylabel(r'$Cp_l - Cp_u$ [-]')
    plt.grid('True')
    plt.legend()

if plot_panel_density:

    plt.figure("Panel Density Convergence")
    plt.title("Panel Density Convergence")
    plt.plot(xxlist[:5]/np.cos(np.deg2rad(12)) + c/4, clx[:5], '-r', label='Npan = 5')
    plt.plot(xxlist[5:25]/np.cos(np.deg2rad(12)) + c/4, clx[5:25], '.b', label='Npan = 20')
    plt.plot(xxlist[25:125]/np.cos(np.deg2rad(12)) + c/4, clx[25:125], '--g', label='Npan = 100')
    plt.ylim(0, 10)
    plt.xlabel('x/c [-]')
    plt.ylabel(r'$C_l$ [-]')
    plt.grid('True')
    plt.legend()

plt.show()