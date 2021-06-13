import numpy as np
import matplotlib.pyplot as plt
import numba as nb


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

    if r_vortex_sq < 1e-9:  # some arbitrary threshold
        return np.array([0.0, 0.0])

    # the distance in x, and z between two points
    dist_vec = np.array([xcol - xvor, zcol - zvor])

    norm_factor = circvor / (2.0 * np.pi * r_vortex_sq)  # circulation at
    # vortex element / circumferential distance

    # induced velocity of vortex element on collocation point
    vel_vor = norm_factor * dcm @ dist_vec

    return vel_vor


def plot_circulatory_loads(alpha_circ, cl_circ, alpha_ss, cl_ss):
    """
    Plots the circulatory and non-circulatory CL as function of the angle of attack, alpha

    :param alpha_circ: corresponding AoA for the circulatory CLs
    :param cl_circ: Array of circulatory CLs
    :param alpha_ss: corresponding AoA for the non-circulatory CLs
    :param cl_ss: Array of non-circulatory (steady-state) CLs
    """

    fig, ax = plt.subplots(1, 1, dpi=150, constrained_layout=True)

    ax.plot(np.degrees(alpha_circ), cl_circ, label=r"Circulatory $C_l$")
    ax.plot(np.degrees(alpha_ss), cl_ss, label=r"Non-circulatory $C_l$")

    ax.grid()
    ax.legend(prop={"size": 12})
    ax.set_xlabel(r"$\alpha$ $[\circ]$", fontsize=12)
    ax.set_ylabel(r"$C_l$ [-]", fontsize=12)


@nb.njit()
def compute_velocity_field_us(u_inf, x_mesh, y_mesh, x_vorts, y_vorts, gamma_b, gamma_w, x_wake, y_wake):
    # Build velocity and pressure distribution
    u = np.ones((len(x_mesh[:, 0]), len(y_mesh[0, :]))) * u_inf
    v = np.zeros((len(x_mesh[:, 0]), len(y_mesh[0, :])))
    v_map = np.zeros((len(x_mesh[:, 0]), len(y_mesh[0, :])))
    cp_map = np.zeros((len(x_mesh[:, 0]), len(y_mesh[0, :])))

    print('...Creating velocity and pressure distribution.\n')

    for i in range(len(x_mesh[:, 0])):

        print('   ...Row:', i + 1)

        for j in range(len(y_mesh[0, :])):

            for g in range(len(gamma_b) - 1):
                uv = lumpvor2d(x_mesh[i, j], y_mesh[i, j], x_vorts[g], y_vorts[g], gamma_b[g])
                u[i, j] = u[i, j] + uv[0]
                v[i, j] = v[i, j] + uv[1]

            for g in range(len(gamma_w)):
                uv = lumpvor2d(x_mesh[i, j], y_mesh[i, j], x_wake[g + 1], y_wake[g + 1], gamma_w[g])
                u[i, j] = u[i, j] + uv[0]
                v[i, j] = v[i, j] + uv[1]

            v_map[i, j] = np.sqrt(u[i, j] ** 2 + v[i, j] ** 2)
            cp_map[i, j] = 1 - (v_map[i, j] / u_inf) ** 2
    return v_map, cp_map


@nb.njit()
def compute_velocity_field_ss(u_inf, x_mesh, y_mesh, x_vorts, y_vorts, gamma):
    # Build velocity and pressure distribution
    u = np.ones((len(x_mesh[:, 0]), len(y_mesh[0, :]))) * u_inf
    v = np.zeros((len(x_mesh[:, 0]), len(y_mesh[0, :])))
    v_map = np.zeros((len(x_mesh[:, 0]), len(y_mesh[0, :])))
    cp_map = np.zeros((len(x_mesh[:, 0]), len(y_mesh[0, :])))

    print('...Creating velocity and pressure distribution.\n')

    for i in range(len(x_mesh[:, 0])):

        print('   ...Row:', i + 1)

        for j in range(len(y_mesh[0, :])):

            for g in range(len(gamma)):
                uv = lumpvor2d(x_mesh[i, j], y_mesh[i, j], x_vorts[g], y_vorts[g], gamma[g])
                u[i, j] = u[i, j] + uv[0]
                v[i, j] = v[i, j] + uv[1]

            v_map[i, j] = np.sqrt(u[i, j] ** 2 + v[i, j] ** 2)
            cp_map[i, j] = 1 - (v_map[i, j] / u_inf) ** 2
    return v_map, cp_map