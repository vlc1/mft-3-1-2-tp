### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ 80d73794-3e10-11eb-1f4c-a703348234ae
using DifferentialEquations

# ╔═╡ 5e554a18-3e55-11eb-201f-77e86a894262
md"""
This notebook is available for download [here](https://github.com/vlc1/mft-3-1-2-tp/blob/master/1.jl).

"""

# ╔═╡ 52ce16ae-3e12-11eb-0819-0538eb4c7ea7
const R̅ = 8.314

# ╔═╡ b0b7d0e6-3b8f-11eb-286e-3bf9f2a16e5e
md"""
# Isothermal, isochoric irreversible kinetics

We consider here one of the simplest physical examples of isothermal, isochoric kinetics : ``\mathrm{O}``-``\mathrm{O}_2`` dissociation and recombination. The ``
N = 2`` compounds are
```math
\mathrm{O} \quad (i = 1)
```
and
```math
\mathrm{O}_2 \quad (i = 2).
```

The ``J = 2`` reactions are
```math
\mathrm{O} + \mathrm{O} + \mathrm{M} \rightarrow \mathrm{O}_2 +  \mathrm{M} \quad (j = 1),
```
```math
\mathrm{O}_2 +  \mathrm{M} \rightarrow \mathrm{O} + \mathrm{O} + \mathrm{M} \quad (j = 2).
```

The irreversible reaction mechanism presented in class (found [here](http://vlc1.github.io/mft-3-1-2/20201124/irreversible.pdf)) provides :
```math
a_1 = 2.90 \times 10 ^ {17} \left ( \frac{\mathrm{mol}}{\mathrm{cm} ^ 3} \right ) ^ {- 2} \frac{\mathrm{K}}{\mathrm{s}}, \quad \beta_1 = -1, \quad \overline{\mathcal{E}}_1 = 0 \frac{\mathrm{kJ}}{\mathrm{mol}}
```
and
```math
a_2 = 6.81 \times 10 ^ {18} \left ( \frac{\mathrm{mol}}{\mathrm{cm} ^ 3} \right ) ^ {- 1} \frac{\mathrm{K}}{\mathrm{s}}, \quad \beta_2 = -1, \quad \overline{\mathcal{E}}_2 = 496.41 \frac{\mathrm{kJ}}{\mathrm{mol}}.
```

Finally, the collision efficiencies are given as ``f_1 = 1`` and ``f_2 = 0.35``.

The temporal evolution of the ``\mathrm{O}`` and ``\mathrm{O}_2`` molar concentrations is given by :
```math
\left \{ \begin{aligned}
\frac{\mathrm{d} \overline{\rho}_1}{\mathrm{d} t} & = -2 r_1 \left ( \overline{\rho}_1, \overline{\rho}_2, T \right ) + 2 r_2 \left ( \overline{\rho}_1, \overline{\rho}_2, T \right ), \\
\frac{\mathrm{d} \overline{\rho}_2}{\mathrm{d} t} & = r_1 \left ( \overline{\rho}_1, \overline{\rho}_2, T \right ) - r_2 \left ( \overline{\rho}_1, \overline{\rho}_2, T \right )
\end{aligned} \right .
```
where
```math
\left \{ \begin{aligned}
r_1 \left ( \overline{\rho}_1, \overline{\rho}_2, T \right ) & = \overline{\rho}_1 \overline{\rho}_1 \overline{\rho}_\mathrm{M} k_1 \left ( T \right ), \\
r_2 \left ( \overline{\rho}_1, \overline{\rho}_2, T \right ) & = \overline{\rho}_2 \overline{\rho}_\mathrm{M} k_2 \left ( T \right )
\end{aligned} \right .
```
and
```math
\overline{\rho}_\mathrm{M} = f_i \overline{\rho}_i,
```
```math
k_i \left ( T \right ) = a_i T ^ {\beta _ i} \exp \left ( - \mathcal{E}_i / \overline{R} T \right ).
```

"""

# ╔═╡ c590997a-3e0e-11eb-0ae6-29ae02a5d277
md"""
**Question** -- Solve this system of equations at ``T = 5000\mathrm{K}`` with the following initial conditions :
```math
\overline{\rho}_1 \left ( 0 \right ) = \overline{\rho}_2 \left ( 0 \right ) = 10 ^ {-3} \mathrm{mol} \cdot \mathrm{cm} ^ {- 3}.
```

The solution will be performed using the [DifferentialEquations.jl package](https://github.com/SciML/DifferentialEquations.jl). A tutorials on how to solve systems of ordinary differential equations using this package may be found [here](https://diffeq.sciml.ai/stable/tutorials/ode_example/#Example-2:-Solving-Systems-of-Equations).

Finally, the temporal evolution of the molar concentrations will be displayed with a logarithmic scale on the ``t``-axis using the `plot` command of the [Plots.jl](https://github.com/JuliaPlots/Plots.jl) by means of the keyword argument `xscale = :log`.

A [variety of solvers](https://diffeq.sciml.ai/stable/solvers/ode_solve/) are available : use them to assess their impact on the numerical solution.

"""

# ╔═╡ b4ef4bc6-3e4b-11eb-33f5-ff7215d0a61e
md"""
**Question** -- Vary the temperature (``T = 300 \mathrm{K}``, ``1000 \mathrm{K}``, ``3000 \mathrm{K}`` and ``10000 \mathrm{K}``) and see how this affects the equilibrium composition.

"""

# ╔═╡ d29493ac-3e4b-11eb-2288-5d7fe5b87ab7
md"""
# Isothermal, isochoric reversible kinetics

We now model this system with a single (``J = 1``) reversible reaction :
```math
\mathrm{O}_2 + \mathrm{M} \leftrightharpoons \mathrm{O} + \mathrm{O} + \mathrm{M}.
```

Here,
```math
a = 1.85 \times 10 ^ {11} \left ( \frac{\mathrm{mol}}{\mathrm{cm} ^ 3} \right ) ^ {- 1} \frac{\mathrm{K} ^ {-0.5}}{\mathrm{s}}, \quad \beta = 0.5, \quad \overline{\mathcal{E}} = 400 \frac{\mathrm{kJ}}{\mathrm{mol}}.
```

The evolution equations are now given by
```math
\left \{ \begin{aligned}
\frac{\mathrm{d} \overline{\rho}_1}{\mathrm{d} t} & = 2 r \left ( \overline{\rho}_1, \overline{\rho}_2, T \right ), \\
\frac{\mathrm{d} \overline{\rho}_2}{\mathrm{d} t} & = -r \left ( \overline{\rho}_1, \overline{\rho}_2, T \right )
\end{aligned} \right .
```
where
```math
r \left ( \overline{\rho}_1, \overline{\rho}_2, T \right ) = k \left ( T \right ) \left ( \overline{\rho}_2 \overline{\rho}_\mathrm{M} - \frac{1}{K_c \left ( T \right )} \overline{\rho}_1 \overline{\rho}_1 \overline{\rho}_\mathrm{M} \right )
```
and
```math
\overline{\rho}_\mathrm{M} = \overline{\rho}_1 + \overline{\rho}_2,
```
```math
k \left ( T \right ) = a T ^ {\beta} \exp \left ( - \mathcal{E} / \overline{R} T \right ).
```

The equilibrium constant is given as a function of temperature
```math
K_c \left ( T \right ) = \frac{P ^ \mathrm{o}}{\overline{R} T} \exp \left ( - \frac{\Delta _r G ^ \mathrm{o} \left ( T \right )}{\overline{R} T} \right )
```
where the standard enthalpy of reaction is given as
```math
\Delta _ r G ^ \mathrm{o} \left ( T \right ) = 2 \overline{G} ^ \mathrm{o} _ 1 \left ( T \right ) - \overline{G} ^ \mathrm{o} _ 2 \left ( T \right )
```
and ``P ^ \mathrm{o} = 1 \, \mathrm{Pa}`` denotes the standard pressure.

**Question** -- Solve this system using the initial conditions used previously for the irreversible reaction.

To do so, you will need to access the thermodynamical properties of ``\mathrm{O}`` and ``\mathrm{O}_2``. This will be achieved by means of the **NASA 9-coefficient polynomial parameterization** (see next section).

"""

# ╔═╡ 590869ac-3ae3-11eb-19c8-d1a75f029e7a
md"""
# NASA 9-coefficient polynomial parameterization

The NASA 9-coefficient polynomial parameterization is used to compute reference-state thermodynamic properties ``\overline{C}_p^\mathrm{o} \left ( T \right )``, ``\overline{H}^\mathrm{o} \left ( T \right )``, ``\overline{S}^\mathrm{o} \left ( T \right )`` and ``\overline{G} ^ \mathrm{o} \left ( T \right )``.

## Definitions

The heat capacity at constant pressure is given as :
```math
\frac{\overline{C}_{p, i}^\mathrm{o} \left ( T \right )}{\overline{R}} = \frac{a_1}{T ^ 2} + \frac{a_2}{T} + a_3 + a_4 T + a_5 T ^ 2 + a_6 T ^ 3 + a_7 T ^ 4.
```

These coefficient also provide the "engineering enthalpy", defined as
```math
\overline{H} ^ \mathrm{o} _i \left ( T \right ) = \Delta_f \overline{H} _i ^ \mathrm{o} \left ( 298,15 \right ) + \int_{298,15}^T \overline{C} ^ \mathrm{o}_{p, i} \left ( \widehat{T} \right ) \mathrm{d} \widehat{T}
```
by the following parameterization
```math
\frac{\overline{H} ^ \mathrm{o} _ i \left ( T \right )}{\overline{R} T} = -\frac{a_1}{T ^ 2} + \frac{a_2 \ln T}{T} + a_3 + \frac{a_4 T}{2} + \frac{a_5 T ^ 2}{3} + \frac{a_6 T ^ 3}{4} + a_7 \frac{T ^ 4}{5} + \frac{b_1}{T}.
```

Finally,
```math
\frac{\overline{S} _i ^ \mathrm{o} \left ( T \right )}{\overline{R}} = -\frac{a_1}{2 T ^ 2} - \frac{a_2}{T} + a_3 \ln T + a_4 T + \frac{a_5 T ^ 2}{2} + \frac{a_6 T ^ 3}{3} + \frac{a_7 T ^ 4}{4} + b_2.
```

Likewise, Gibbs' free energy can be obtained from
```math
\begin{aligned}
\frac{\overline{G} ^ \mathrm{o} _ i \left ( T \right )}{\overline{R} T} & = \frac{\overline{H} ^ \mathrm{o} _ i \left ( T \right )}{\overline{R} T} - \frac{\overline{S} ^ \mathrm{o} _ i \left ( T \right )}{\overline{R}}, \\
& = -\frac{a_1}{2 T ^2} + \frac{2 a_2 \left ( 1 - \ln T \right )}{T} + a_3 \left ( 1 - \ln T \right ) - \frac{a_4 T}{2} - \frac{a_5 T ^ 2}{6} - \frac{a_6 T^3}{12} - \frac{a_7 T ^ 4}{20} + \frac{b_1}{T} - b_2.
\end{aligned}
```

**Question** -- Using this [database](https://shepherd.caltech.edu/EDL/PublicResources/sdt/SDToolbox/cti/NASA9/nasa9.dat) and the [description of the format](https://shepherd.caltech.edu/EDL/PublicResources/sdt/formats/nasa.html), write a function that returns the standard enthalpy of reaction of the reversible dissociation equation (``\Delta_r \overline{G} ^ \mathrm{o}``) as a function of ``T``.

"""

# ╔═╡ 4fd5e632-3e50-11eb-0eb7-d70b9e97c395
md"""
# Going further

## Adiabatic, isochoric reversible kinetics

The isothermal assumption is replaced by an adiabatic one. Temperature now becomes a dependent variable whose evolution is governed by
```math
\overline{\rho} \overline{C} _ V \frac{\mathrm{d} T}{\mathrm{d} t} = -r \Delta_r E ^ \mathrm{o} \left ( T \right )
```
where ``\overline{\rho}`` and ``\overline{C} _ V`` denote the molar concentration and heat capacity at constant volume of the mixture, while
```math
\Delta_r E ^ \mathrm{o} \left ( T \right ) = 2 \overline{E} ^ \mathrm{o} _ 2 \left ( T \right ) - \overline{E} ^ \mathrm{o} _ 1 \left ( T \right ).
```

**Question** -- Solve this problem numerically.

## Zel'dovich mechanism of ``\mathrm{NO}`` production

``\mathrm{NO}`` is one of the major pollutant from combustion processes. It is most important for high-temperature applications.

Zel'dovich mechanism of ``\mathrm{NO}`` production is
```math
(j = 1) \quad \mathrm{N} + \mathrm{NO} \leftrightharpoons \mathrm{N} _ 2 + \mathrm{O},
```
```math
(j = 2) \quad \mathrm{N} + \mathrm{O} _ 2 \leftrightharpoons \mathrm{NO} + \mathrm{O}.
```

This mechanism has ``J = 2`` reactions and ``N = 5`` species : ``\mathrm{NO}`` (``i = 1``), ``\mathrm{N}`` (``i = 2``), ``\mathrm{N}_2`` (``i = 3``) ``\mathrm{O}`` (``i = 4``) and ``\mathrm{O}_2`` (``i = 5``).

The parameters of the chemical reaction network are given as
```math
a_1 = 2.107 \times 10 ^ {13} \left ( \frac{\mathrm{mol}}{\mathrm{cm} ^ 3} \right ) ^ {- 1} \mathrm{s} ^ {- 1}, \quad \beta_1 = 0, \quad \overline{\mathcal{E}}_1 = 0 \frac{\mathrm{kJ}}{\mathrm{mol}}
```
and
```math
a_2 = 5.8394 \times 10 ^ {9} \left ( \frac{\mathrm{mol}}{\mathrm{cm} ^ 3} \right ) ^ {- 1} \frac{\mathrm{K} ^ {-1.01}}{\mathrm{s}}, \quad \beta_2 = 1.01, \quad \overline{\mathcal{E}}_2 = 25.94 \frac{\mathrm{kJ}}{\mathrm{mol}}.
```

**Question** -- Solve for the isothermal and isochoric evolution of this mixture with the initial conditions
```math
\overline{\rho}_1 \left ( 0 \right ) = \overline{\rho}_2 \left ( 0 \right ) = \cdots = \overline{\rho}_5 \left ( 0 \right ) = 10 ^ {- 6} \mathrm{mol} \cdot \mathrm{cm} ^ {-3}
```
and
```math
T = 6000 \mathrm{K}.
```

"""

# ╔═╡ Cell order:
# ╟─5e554a18-3e55-11eb-201f-77e86a894262
# ╠═52ce16ae-3e12-11eb-0819-0538eb4c7ea7
# ╠═80d73794-3e10-11eb-1f4c-a703348234ae
# ╟─b0b7d0e6-3b8f-11eb-286e-3bf9f2a16e5e
# ╟─c590997a-3e0e-11eb-0ae6-29ae02a5d277
# ╟─b4ef4bc6-3e4b-11eb-33f5-ff7215d0a61e
# ╟─d29493ac-3e4b-11eb-2288-5d7fe5b87ab7
# ╟─590869ac-3ae3-11eb-19c8-d1a75f029e7a
# ╟─4fd5e632-3e50-11eb-0eb7-d70b9e97c395
