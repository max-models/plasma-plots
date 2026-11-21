"""MHD wave speeds

Phase speed of the shear Alfvén, slow and fast magnetosonic waves against the angle to the magnetic field.

Ideal MHD has three wave branches. Along the field the shear Alfvén wave travels at $v_A$ and
the others at $v_A$ and $c_s$; across it only the fast wave propagates, at $\\sqrt{v_A^2 + c_s^2}$.
Here $c_s = 0.5\\,v_A$.
"""

import matplotlib.pyplot as plt
import numpy as np

from plasma_plots.theory import waves

theta = np.linspace(0.0, np.pi / 2, 200)
branches = waves.mhd_waves(1.0, theta=theta, alfven_speed=1.0, sound_speed=0.5)

fig = plt.figure(figsize=(9, 4), layout="constrained")
ax = fig.add_subplot(1, 2, 1)
for name, omega in branches.items():
    ax.plot(np.degrees(theta), omega.real, lw=2, label=name)
ax.set(xlabel="angle to B [deg]", ylabel=r"$\omega/(k v_A)$", title="Phase speed")
ax.legend(frameon=False)

polar = fig.add_subplot(1, 2, 2, projection="polar")
for name, omega in branches.items():
    polar.plot(
        np.r_[theta, -theta[::-1]],
        np.r_[omega.real, omega.real[::-1]],
        lw=2,
        label=name,
    )
polar.set_thetamin(-90)
polar.set_thetamax(90)
polar.set_title("Friedrichs diagram")

plt.show()
