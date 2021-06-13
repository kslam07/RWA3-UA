import numpy as np
import matplotlib.pyplot as plt

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
    ax.legend()
    ax.set_xlabel(r"$\alpha$ $[\circ]$")
    ax.set_ylabel(r"$C_l$ [-]")
