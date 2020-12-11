### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ 376b38ea-3aef-11eb-009e-b59d8420aa7b
using Plots

# ╔═╡ 4ba3e394-3ae9-11eb-0541-5fb7b933ee53
const R = 8.314

# ╔═╡ 590869ac-3ae3-11eb-19c8-d1a75f029e7a
md"""
# NASA 9-coefficient polynomial parameterization

The NASA 9-coefficient polynomial parameterization is used to compute reference-state thermodynamic properties ``C_p^\mathrm{o} \left ( T \right )``, ``H^\mathrm{o} \left ( T \right )`` and ``S^\mathrm{o} \left ( T \right )``.

## Definitions

The heat capacity at constant pressure is given as :
```math
\frac{C_p^\mathrm{o} \left ( T \right )}{\overline{R}} = \frac{a_1}{T ^ 2} + \frac{a_2}{T} + a_3 + a_4 T + a_5 T ^ 2 + a_6 T ^ 3 + a_7 T ^ 4.
```

These coefficient also provide the "engineering enthalpy", defined as
```math
H^\mathrm{o} \left ( T \right ) = \Delta_f H^\mathrm{o} \left ( 298,15 \right ) + \int_{298,15}^T C^\mathrm{o}_p \left ( \widehat{T} \right ) \mathrm{d} \widehat{T}
```
by the following parameterization
```math
\frac{H^\mathrm{o} \left ( T \right )}{\overline{R} T} = -\frac{a_1}{T ^ 2} + \frac{a_2 \ln T}{T} + a_3 + \frac{a_4 T}{2} + \frac{a_5 T ^ 2}{3} + \frac{a_6 T ^ 3}{4} + a_7 \frac{T ^ 4}{5} + \frac{b_1}{T}.
```

Finally,
```math
\frac{S^\mathrm{o} \left ( T \right )}{\overline{R}} = -\frac{a_1}{2 T ^ 2} - \frac{a_2}{T} + a_3 \ln T + a_4 T + \frac{a_5 T ^ 2}{2} + \frac{a_6 T ^ 3}{3} + \frac{a_7 T ^ 4}{4} + b_2.
```

Likewise, Gibbs' free energy can be obtained from
```math
\begin{aligned}
\frac{G^\mathrm{o} \left ( T \right )}{\overline{R} T} & = \frac{H^\mathrm{o} \left ( T \right )}{\overline{R} T} - \frac{S^\mathrm{o} \left ( T \right )}{\overline{R}}, \\
& = -\frac{a_1}{2 T ^2} + \frac{2 a_2 \left ( 1 - \ln T \right )}{T} + a_3 \left ( 1 - \ln T \right ) - \frac{a_4 T}{2} - \frac{a_5 T ^ 2}{6} - \frac{a_6 T^3}{12} - \frac{a_7 T ^ 4}{20} + \frac{b_1}{T} - b_2.
\end{aligned}
```

These quantities can be combined to find *e.g.* 
More information can be found [here](http://garfield.chem.elte.hu/Burcat/burcat.html).

* [Brief description of format](https://shepherd.caltech.edu/EDL/PublicResources/sdt/formats/nasa.html)
* [Example of generated database](https://shepherd.caltech.edu/EDL/PublicResources/sdt/SDToolbox/cti/NASA9/nasa9.dat).

"""

# ╔═╡ 864ffc5e-3ae8-11eb-06af-0d6656f132b9
md"""
## Implementation

"""

# ╔═╡ 8e9aff9e-3ae8-11eb-3b3e-1789ae6457cf
begin
	struct NASAPoly{T}
		a::NTuple{7,T}
		b::NTuple{2,T}
		Δ::T
	end

	getcoef(x) = x.a, x.b, x.Δ
end

# ╔═╡ b59db3ac-3ae8-11eb-3ba4-55696930e207
function Cₚᵒ(p, T)
	a, _ = getcoef(p)
	R * (
		+ a[1] / T ^ 2
		+ a[2] / T
		+ a[3]
		+ a[4] * T
		+ a[5] * T ^ 2
		+ a[6] * T ^ 3
		+ a[7] * T ^ 4
	)
end

# ╔═╡ 401e3b50-3aee-11eb-301f-c1f9c3f4d0eb
function Hᵒ(p, T)
	a, b, _ = getcoef(p)
	R * T * (
		- a[1] / T ^ 2
		+ a[2] * log(T) / T
		+ a[3]
		+ a[4] * T / 2
		+ a[5] * T ^ 2 / 3
		+ a[6] * T ^ 3 / 4
		+ a[7] * T ^ 4 / 5
		+ b[1] / T
	)
end

# ╔═╡ dd084262-3aee-11eb-19e8-a3cfbfd752d1
function Sᵒ(p, T)
	a, b = getcoef(p)
	R * (
		- a[1] / (2T ^ 2)
		- a[2] / T
		+ a[3] * log(T)
		+ a[4] * T
		+ a[5] * T ^ 2 / 2
		+ a[6] * T ^ 3 / 3
		+ a[7] * T ^ 4 / 4
		+ b[2]
	)
end

# ╔═╡ 9ece2244-3aeb-11eb-1837-f76d8ed4aa2f
Ar = NASAPoly(
	(0., 0., 2.5, 0., 0.0, 0., 0.),
	(-7.45375e2, 4.37967491),
	6197.428
)

# ╔═╡ 6cd15892-3af0-11eb-19b1-8f2dcb2a6658
H₂O = NASAPoly(
	(-3.947960830e4, 5.75573102e2, 9.31782653e-1, 7.22271286e-3, -7.34255737e-6, 4.95504349e-9, -1.336933246e-12),
	(-3.30397431e4, 1.724205775e1),
	9904.092
)

# ╔═╡ 7e693d34-3af2-11eb-0e21-1f5dc94e1ffc
O₂ = NASAPoly(
	(-3.42556342e4, 4.84700097e2, 1.119010961, 4.29388924e-3, -6.83630052e-7, -2.0233727e-9, 1.039040018e-12),
	(-3.39145487e3, 1.84969947e1),
	8680.104
)

# ╔═╡ 1fffd6e4-3aee-11eb-22a4-8536781336e6
Cₚᵒ(H₂O, 298.15)

# ╔═╡ 9678348a-3aee-11eb-16bb-2788d16ee910
Hᵒ(O₂, 298.15)

# ╔═╡ 2b361cac-3aef-11eb-2494-c5c66cc051d8
Sᵒ(Ar, 100.)

# ╔═╡ 3b632606-3aef-11eb-1d2c-89f35654baee
begin
	plot(x -> Sᵒ(Ar, x), xlim = (200, 1000))
end

# ╔═╡ 2a90e3f2-3aee-11eb-2884-55a934d2d8a3
5/2*R

# ╔═╡ Cell order:
# ╠═4ba3e394-3ae9-11eb-0541-5fb7b933ee53
# ╟─590869ac-3ae3-11eb-19c8-d1a75f029e7a
# ╠═864ffc5e-3ae8-11eb-06af-0d6656f132b9
# ╠═8e9aff9e-3ae8-11eb-3b3e-1789ae6457cf
# ╠═b59db3ac-3ae8-11eb-3ba4-55696930e207
# ╠═401e3b50-3aee-11eb-301f-c1f9c3f4d0eb
# ╠═dd084262-3aee-11eb-19e8-a3cfbfd752d1
# ╠═9ece2244-3aeb-11eb-1837-f76d8ed4aa2f
# ╠═6cd15892-3af0-11eb-19b1-8f2dcb2a6658
# ╠═7e693d34-3af2-11eb-0e21-1f5dc94e1ffc
# ╠═1fffd6e4-3aee-11eb-22a4-8536781336e6
# ╠═9678348a-3aee-11eb-16bb-2788d16ee910
# ╠═2b361cac-3aef-11eb-2494-c5c66cc051d8
# ╠═376b38ea-3aef-11eb-009e-b59d8420aa7b
# ╠═3b632606-3aef-11eb-1d2c-89f35654baee
# ╠═2a90e3f2-3aee-11eb-2884-55a934d2d8a3
