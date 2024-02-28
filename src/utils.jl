function get_path(dj::DijkstraState, v::Integer)
    u = v
    path = [u]
    while dj.parents[u] != 0
        u = dj.parents[u]
        pushfirst!(path, u)
    end
    return path
end

function torus(n::Int=1000, noise::Real=0.03; segments=1, rng::AbstractRNG=default_rng())
    θ = 2π*rand(rng, n)
    ϕ = 2π*rand(rng, n)
    rs = 0.1
    rb = 0.4
    X = fill(0.0, 3, n)
    R = (rb .+ rs.*cos.(ϕ))
    X[1,:] = R.*cos.(θ)
    X[2,:] = R.*sin.(θ)
    X[3,:] = rs.*sin.(ϕ)
    X .+ noise*randn(3,n)
    mn,mx = extrema(R)
    labels = segments == 0 ? R : round.(Int, (R.-mn)./(mx-mn).*(segments-1))
    X,labels
end
