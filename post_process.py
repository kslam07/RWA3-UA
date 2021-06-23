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


def plot_circulatory_loads(alpha_circ, dalpha_circ, cl_circ, cl_ss, x_wake, y_wake, u_inf, c, time_arr, plot_wake=False):
    """
    Plots the circulatory and non-circulatory CL as function of the angle of attack, alpha
    :param alpha_circ: corresponding AoA for the circulatory CLs
    :param dalpha_circ: diervative of alpha
    :param cl_circ: Array of circulatory CLs
    :param alpha_ss: corresponding AoA for the non-circulatory CLs
    :param cl_ss: Array of non-circulatory (steady-state) CLs
    """

    fig, ax = plt.subplots(1, 2, dpi=150, constrained_layout=True, sharey=True)

    # compute quasi-steady CL
    alpha_qs = (alpha_circ + c / (2 * u_inf) * dalpha_circ)
    cl_qs = 2 * np.pi * alpha_qs

    # only plot 1 period of unsteady CL and steady CL
    idx = np.where(alpha_circ[1:] * alpha_circ[:-1] < 0)[0][2] + 1
    idx = None
    # unsteady CL
    ax[0].plot(np.degrees(alpha_circ)[:idx], cl_circ[:idx], label=r"Unsteady $C_l$", linestyle='-.')
    ax[1].plot(np.degrees(alpha_circ)[:idx], cl_circ[:idx], label=r"Unsteady $C_l$", linestyle='-.')
    # steady CL
    ax[0].plot(np.degrees(alpha_circ)[:idx], cl_ss[:idx], label=r"Steady $C_l$", c='r')
    # quasi-steady CL
    ax[1].plot(np.degrees(alpha_circ), cl_qs, label=r"Quasi-steady $C_l$", c='r', linestyle='--')

    ax[0].grid()
    ax[1].grid()
    ax[0].legend(prop={"size": 14}, loc="lower right")
    ax[1].legend(prop={"size": 14}, loc="lower right")
    ax[0].set_ylabel(r"$C_l$ [-]", fontsize=14)
    ax[0].set_xlabel(r"$\alpha$ $[\circ]$", fontsize=14)
    ax[1].set_xlabel(r"$\alpha$ $[\circ]$", fontsize=14)

    if plot_wake:
        fig, ax = plt.subplots(1, 1, dpi=150, constrained_layout=True)
        ax.scatter(x_wake, y_wake, label="Wake vortices")
        ax.grid()
        ax.set_xlabel("Horizontal distance [m]", fontsize=14)
        ax.set_ylabel("Vertical distance [m]", fontsize=14)
        ax.legend(prop={"size": 14}, loc=1)

        # compute equivalent AoA
        x_lag_old = 0
        y_lag_old = 0
        a1 = 0.165
        a2 = 0.335
        b1 = 0.045
        b2 = 0.3
        dt = 0.1
        idx = np.where(alpha_circ[1:] * alpha_circ[:-1] < 0)[0][1] + 1
        dalpha_qs = alpha_qs[1:] - alpha_qs[:-1]
        alpha_e = []
        for i, alpha_curr in enumerate(alpha_qs[1:]):
            ds = 2 * dt * u_inf / c
            x_lag = x_lag_old * np.exp(-b1 * ds) + dalpha_qs[i] * a1 * np.exp(-b1 * ds / 2)
            y_lag = y_lag_old * np.exp(-b2 * ds) + dalpha_qs[i] * a2 * np.exp(-b2 * ds / 2)
            alpha_e.append(alpha_curr - x_lag - y_lag)

            x_lag_old = x_lag
            y_lag_old = y_lag

        # compute s
        s = 2 * time_arr * u_inf / c

        fig, ax = plt.subplots(1, 1, dpi=150, constrained_layout=True)
        ax.plot(s[1:idx], np.degrees(alpha_e)[1:idx], label="Unsteady", linestyle='-.')
        ax.plot(s[1:idx], np.degrees(alpha_qs)[1:idx], label="Quasi-steady", linestyle='--')
        ax.plot(s[1:idx], np.degrees(alpha_circ)[1:idx], label="Steady", c='r')
        ax.grid()
        ax.legend(prop={"size": 14})
        ax.set_xlabel("semi-chord s [-]", fontsize=14)
        ax.set_ylabel(r"$\alpha$ [$\circ$]", fontsize=14)

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