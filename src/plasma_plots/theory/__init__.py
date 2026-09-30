"""Analytic theory to compare Struphy runs against.

Dispersion relations, plasma parameters, orbits, exact solutions and the properties of the
numerical schemes.

Everything here is plain numpy (no scipy) and independent of Struphy. The modules:

* :mod:`~plasma_plots.theory.special`: the Faddeeva and plasma dispersion functions, complete
  elliptic integrals.
* :mod:`~plasma_plots.theory.parameters`: plasma frequencies, lengths, speeds and dimensionless
  numbers, in SI units.
* :mod:`~plasma_plots.theory.kinetic`: kinetic dispersion relations: Langmuir waves and Landau
  damping, ion-acoustic waves, beam-plasma, bump-on-tail and two-stream instabilities, Weibel.
* :mod:`~plasma_plots.theory.waves`: fluid, MHD and cold-plasma waves: light waves, MHD waves at
  any angle, dissipative and Hall-MHD Alfvén waves, cold-plasma (Stix) waves, cutoffs and
  resonances, Faraday rotation, cavity modes, drift waves and Hasegawa–Wakatani, Alfvén and slow
  continua and the TAE frequency.
* :mod:`~plasma_plots.theory.orbits`: gyromotion, guiding-center drifts, trapped particles,
  bounce frequencies and banana widths in a large-aspect-ratio tokamak.
* :mod:`~plasma_plots.theory.exact`: exact solutions to verify simulations: the Riemann problem
  of gas dynamics (with vacuum), dam break, diffusion, advection, pressureless flow and its
  caustics.
* :mod:`~plasma_plots.theory.numerics`: accuracy of the numerics: amplification and phase errors
  of time integrators, and the numerical dispersion of spline finite elements.

Conventions
-----------
* **Arrays**: every function takes scalars or numpy arrays that broadcast against each other,
  and returns numpy arrays (a Python scalar for scalar input).
* **Units**: normalized units unless a function says otherwise. Kinetic results are in units of
  the plasma frequency ω_p (time), the Debye length λ_D (length) and the thermal speed
  v_th = √(T/m) (velocity); fluid and MHD results in whatever units the speeds and wavenumbers
  are given in. :mod:`~plasma_plots.theory.parameters` works in SI units.
* **Frequencies** are complex for ``exp(i(k·x − ωt))``: the real part is the frequency, a positive
  imaginary part a growth rate, a negative one a damping rate.
* **Several branches** come as a dict of branch names to complex frequencies, e.g.
  ``{"shear Alfvén": ..., "slow": ..., "fast": ...}``. Such a function (or any callable that
  returns such a dict, including Struphy's own ``struphy.dispersion_relations`` objects) can be
  passed to ``array.plasma.plot.dispersion(branches=...)`` as it is; for one branch, a lambda:
  ``branches={"fast": lambda k: mhd_waves(k, ...)["fast"]}``.

Examples
--------
The Landau damping of a Langmuir wave at k λ_D = 0.5, against a Vlasov–Ampère run's measured
frequency and damping rate:

>>> from plasma_plots.theory import kinetic
>>> omega = kinetic.langmuir(0.5)
>>> round(omega.real, 4), round(omega.imag, 4)
(1.4157, -0.1534)
"""
