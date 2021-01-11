### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ 15aa43a2-540a-11eb-3250-99ccb0d9adb3
using DifferentialEquations, Plots

# ╔═╡ 879492d4-53f0-11eb-3b6d-7b4f33684e2a
md"""
# Numerical resolution of Boundary Value Problems (BVP)

A general way of expressing write BVP is
```math
\left \{ \begin{aligned}
F \left ( \dot{u}, u, t \right ) & = 0, \\
h \left ( u \left ( a \right ), u \left ( b \right ) \right ) & = 0
\end{aligned} \right .
```
where the first and second equations are respectively referred to as the *governing equation* (solved over ``\left [ a, b \right ]``) and the *boundary condition*. ``t`` is a temporal or spatial coordinate referred to as the independent variable, and ``u`` is a ``\mathbb{R} ^ n``-valued field referred to as the dependent variable (``n \in \mathbb{N} ^ *``).

A necessary condition for the BVP to be well-posed is for the jacobian of the function
```math
h \colon \left \vert \begin{array}{cll}
\mathbb{R} ^ n \times \mathbb{R} ^ n & \to & \mathbb{R} ^ n, \\
\left ( x, y \right ) & \mapsto & h \left ( x, y \right )
\end{array} \right .
```
to have full rank.

In addition, when the jacobian of the function
```math
F \colon \left \vert \begin{array}{cll}
\mathbb{R} ^ n \times \mathbb{R} ^ n \times \left [ a, b \right ] & \to & \mathbb{R} ^ n, \\
\left ( x, y, t \right ) & \mapsto & F \left ( x, y, t \right )
\end{array} \right .
```
with respect to the first argument (``x``) has full rank, a less general albeit more practical way to recast the governing equation is as follows
```math
\dot{u} = f \left ( u, t \right ).
```

There exists a variety of techniques for numerically solving BVP, and we will consider two :

* Shooting method,
* Collocation method.

# Steady conduction equation

Let us consider as a first example the steady conduction of heat
```math
0 = \frac{1}{r ^ \alpha} \frac{\mathrm{d}}{\mathrm{d} r} \left ( r ^ \alpha \lambda \frac{\mathrm{d} T}{\mathrm{d} r} \right ).
```
Here, the independent coordinate ``t = r`` refers to a Cartesian (``\alpha = 0``), cylindrical (``\alpha = 1``) or spherical (``\alpha = 2``) coordinates.

**Questions**

1. Rewrite this second order Ordinary Differential Equation (ODE) in the form of a system of coupled first order ODEs.
1. Write the function ``h`` corresponding to the boundary conditions reported below and show that its jacobian has full rank.
```math
\left \{ \begin{aligned}
T \left ( a \right ) & = 0 \\
\frac{\mathrm{d} T}{\mathrm{d} r} \left ( b \right ) & = 1
\end{aligned} \right .
```
3. Solve this BVP numerically for ``\alpha = 2`` (sphere), ``a = 1`` and ``b = 10`` and compare the result to the analytical solution.

!!! tip "Tip"

Check `DifferentialEquations.jl`'s [documentation on BVPs](https://diffeq.sciml.ai/stable/tutorials/bvp_example/).

"""

# ╔═╡ 7295c0c0-540d-11eb-1530-173db3a04a20
md"""
# Steady convection-diffusion equation

We model steady oxidisation of char particle by the following reactions : direct oxidisation of char at the particle surface
```math
\mathrm{C} \left [ s \right ] + \frac{1}{2} \mathrm{O}_2 \left [ g \right ] \rightarrow \mathrm{CO} \left [ g \right ],
```
and the gas phase, the one-step global reaction
```math
2 \mathrm{CO} \left [ g \right ] + \mathrm{O}_2 \left [ g \right ] \rightarrow 2\mathrm{CO}_2 \left [ g \right ].
```

The following non-dimensional set of equations models the steady direct oxidisation of a coal particle in an oxy-atmosphere :
```math
\left \{ \begin{aligned}
\frac{1}{r ^ \alpha} \frac{\mathrm{d} m}{\mathrm{d} r} & = 0, \\
\frac{m}{r ^ \alpha} \frac{\mathrm{d} Y_1}{\mathrm{d} r} - \frac{1}{r ^ \alpha} \frac{\mathrm{d}}{\mathrm{d} r} \left ( r ^ \alpha \frac{\mathrm{d} Y_1}{\mathrm{d} r} \right ) & = -\frac{5}{7} \Omega \left ( Y_1, Y_2, T \right ), \\
\frac{m}{r ^ \alpha} \frac{\mathrm{d} Y_2}{\mathrm{d} r} - \frac{1}{r ^ \alpha} \frac{\mathrm{d}}{\mathrm{d} r} \left ( r ^ \alpha \frac{\mathrm{d} Y_2}{\mathrm{d} r} \right ) & = -\frac{2}{7} \Omega \left ( Y_1, Y_2, T \right ), \\
\frac{m}{r ^ \alpha} \frac{\mathrm{d} Y_3}{\mathrm{d} r} - \frac{1}{r ^ \alpha} \frac{\mathrm{d}}{\mathrm{d} r} \left ( r ^ \alpha \frac{\mathrm{d} Y_3}{\mathrm{d} r} \right ) & = \Omega \left ( Y_1, Y_2, T \right ), \\
\frac{m}{r ^ \alpha} \frac{\mathrm{d} T}{\mathrm{d} r} - \frac{1}{r ^ \alpha} \frac{\mathrm{d}}{\mathrm{d} r} \left ( r ^ \alpha \frac{\mathrm{d} T}{\mathrm{d} r} \right ) & = \Omega \left ( Y_1, Y_2, T \right )
\end{aligned} \right .
```
where ``\alpha = 2``, ``Y_1``, ``Y_2`` and ``Y_3`` respectively denote the carbon monoxyde, oxygen and carbon dioxyde mass fractions and
```math
\Omega \colon \left ( Y_1, Y_2, T \right ) \mapsto \mathrm{Da}_g \frac{Y_1 ^ 2 Y_2}{T ^ 2} \exp \left ( - \frac{\theta}{T} \right ).
```

The boundary conditions are set to
```math
Y_1 \left ( a \right ) + Y_2 \left ( a \right ) + Y_3 \left ( a \right ) = 1,
```
```math
\begin{aligned}
m \left ( a \right ) Y_1 \left ( a \right ) - Y_1' \left ( a \right ) & = \frac{5}{2} \omega \left ( Y_2 \left ( a \right ) \right ), \\
m \left ( a \right ) Y_2 \left ( a \right ) - Y_2' \left ( a \right ) & = -\omega \left ( Y_2 \left ( a \right ) \right ), \\
m \left ( a \right ) Y_3 \left ( a \right ) - Y_3' \left ( a \right ) & = 0,
\end{aligned}
```
and
```math
T \left ( a \right ) = T ^ s
```
where ``a = 1`` denote the particle surface, and
```math
\begin{aligned}
Y_1 \left ( b \right ) & = Y _ 1 ^ \infty, \\
Y_2 \left ( b \right ) & = Y _ 2 ^ \infty, \\
Y_3 \left ( b \right ) & = Y _ 3 ^ \infty,
\end{aligned}
```
and
```math
T \left ( b \right ) = T ^ \infty
```
where ``b = \infty`` denote the free-stream.

The function ``\omega`` is defined as
```math
\omega \colon \left ( Y_2 \right ) \mapsto \mathrm{Da} _ s Y _ 2.
```

**Questions**

1. Write the system in first order form.
1. Show that the boundary conditions are well-posed.
1. Define and implement the functions ``f`` and ``h`` that correspond to this system.

"""

# ╔═╡ Cell order:
# ╠═15aa43a2-540a-11eb-3250-99ccb0d9adb3
# ╟─879492d4-53f0-11eb-3b6d-7b4f33684e2a
# ╟─7295c0c0-540d-11eb-1530-173db3a04a20
