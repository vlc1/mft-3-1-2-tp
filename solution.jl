### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ 80d73794-3e10-11eb-1f4c-a703348234ae
using DifferentialEquations

# ╔═╡ 9636997a-3e17-11eb-0d63-3d670ba29f15
using StaticArrays

# ╔═╡ 376b38ea-3aef-11eb-009e-b59d8420aa7b
using Plots

# ╔═╡ 5e554a18-3e55-11eb-201f-77e86a894262


# ╔═╡ 52ce16ae-3e12-11eb-0819-0538eb4c7ea7
const R̅ = 8.314

# ╔═╡ 9aed6c00-3b92-11eb-141c-97857b501dd1
struct IrrMech{J,N,T,U<:SVector{J,T},V<:SVector{N,T},W<:SMatrix{N,J,T}}
	a::U
	β::U
	ℰ::U
	f::V
	νf::W
	νb::W
	νm::U
end

# ╔═╡ 6bf5bdc0-3b98-11eb-1aa8-1d6293d332b4
flatten(mech::IrrMech) = mech.a, mech.β, mech.ℰ, mech.f, mech.νf, mech.νb, mech.νm

# ╔═╡ af501f42-3e19-11eb-1600-1b85882f8b8e
reactions(::IrrMech{J}) where {J} = SOneTo{J}()

# ╔═╡ 9c4c1344-3e19-11eb-3b69-dbbca45c9fb9
compounds(::IrrMech{J,N}) where {J,N} = SOneTo{N}()

# ╔═╡ c3ffe446-3b93-11eb-3472-e13df5c1af25
mech = IrrMech(
	SVector(2.90e17, 6.81e18),
	SVector(-1., -1.),
	SVector(0., 496.41),
	SVector(1., 0.35),
	SMatrix{2,2}(2., 0., 0., 1.),
	SMatrix{2,2}(0., 1., 2., 0.),
	SVector(1., 1.)
)

# ╔═╡ ba1508aa-3e0f-11eb-1dab-7da1fa840b9c
function dissociation!(dρ̅, ρ̅, (mech, θ), t)
	a, β, ℰ, f, νf, νb, νm = flatten(mech)

	ρ̅ₘ = sum(f .* ρ̅)

	k = @. a * θ ^ β * exp(-ℰ / R̅ / θ)
	#=r = map(reactions(mech)) do j
		prod(ρ̅ .^ νf[j]) * ρ̅ₘ ^ νm[j] * k[j]
	end
	=#

	#=
	dρ̅ .= map(1:N) do i
		νf
	end
	=#
	dρ̅[1] = -ρ̅[1]
	dρ̅[2] = -ρ̅[2]
end

# ╔═╡ 72e8c934-3e12-11eb-38ea-a53840ae1794
a, β, ℰ, f, νf, νb, νm = flatten(mech)

# ╔═╡ f3ef6ece-3e13-11eb-220e-559b0de536cd
νf[1]

# ╔═╡ b925eb4a-3e12-11eb-2047-57e7d07b5970


# ╔═╡ 29f28d02-3e10-11eb-09ef-e1c74931c1f3
begin
	ρ̅₀ = SVector(1e-3, 1e-3)
	θ = 5e3
	span = (0e0, 1e-6)
end

# ╔═╡ 154340be-3e19-11eb-1ad7-95e70a798193
prod.(ρ̅₀ .^ νf)

# ╔═╡ 08e16b9c-3e1a-11eb-20fd-4fd9ee1ff2c6
k = @. a * θ ^ β * exp(-ℰ / R̅ / θ)

# ╔═╡ e6e18f14-3e13-11eb-06e9-a195d177747a
ρ̅ₘ = sum(f .* ρ̅₀)

# ╔═╡ f8bcc746-3e19-11eb-0867-11e009da2a88
r = map(reactions(mech)) do j
		prod(ρ̅₀ .^ νf[j]) * ρ̅ₘ ^ νm[j] * k[j]
	end

# ╔═╡ e33d9f10-3e13-11eb-1755-3fea411c8770
prod((ρ̅₀..., ρ̅ₘ) .^ νf[1])

# ╔═╡ 75ed924c-3e10-11eb-09c5-87156320acdf
prob = ODEProblem(dissociation!, ρ̅₀, span, (mech, θ))

# ╔═╡ 7bd247a2-3e10-11eb-3a05-956b20098e7c
sol = solve(prob)

# ╔═╡ 087e074e-3e12-11eb-231c-fb94dd2d2dbf
plot(sol)

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
# ╟─5e554a18-3e55-11eb-201f-77e86a894262
# ╠═52ce16ae-3e12-11eb-0819-0538eb4c7ea7
# ╠═80d73794-3e10-11eb-1f4c-a703348234ae
# ╠═9636997a-3e17-11eb-0d63-3d670ba29f15
# ╠═9aed6c00-3b92-11eb-141c-97857b501dd1
# ╠═6bf5bdc0-3b98-11eb-1aa8-1d6293d332b4
# ╠═af501f42-3e19-11eb-1600-1b85882f8b8e
# ╠═9c4c1344-3e19-11eb-3b69-dbbca45c9fb9
# ╠═c3ffe446-3b93-11eb-3472-e13df5c1af25
# ╠═ba1508aa-3e0f-11eb-1dab-7da1fa840b9c
# ╠═72e8c934-3e12-11eb-38ea-a53840ae1794
# ╠═154340be-3e19-11eb-1ad7-95e70a798193
# ╠═08e16b9c-3e1a-11eb-20fd-4fd9ee1ff2c6
# ╠═f8bcc746-3e19-11eb-0867-11e009da2a88
# ╠═e6e18f14-3e13-11eb-06e9-a195d177747a
# ╠═e33d9f10-3e13-11eb-1755-3fea411c8770
# ╠═f3ef6ece-3e13-11eb-220e-559b0de536cd
# ╠═b925eb4a-3e12-11eb-2047-57e7d07b5970
# ╠═29f28d02-3e10-11eb-09ef-e1c74931c1f3
# ╠═75ed924c-3e10-11eb-09c5-87156320acdf
# ╠═7bd247a2-3e10-11eb-3a05-956b20098e7c
# ╠═087e074e-3e12-11eb-231c-fb94dd2d2dbf
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
