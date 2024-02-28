# Parallel Transport Unfolding (PTU)
# ---------------------------
# Budninskiy, Max, Glorian Yin, Leman Feng, Yiying Tong, and Mathieu Desbrun. 
# “Parallel Transport Unfolding: A Connection-Based Manifold Learning Approach.” arXiv, November 2, 2018.
# http://arxiv.org/abs/1806.09039.


"""
    PTU{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction

The `PTU type represents a local tangent space alignment model constructed for `T` type data with a help of the `NN` nearest neighbor algorithm.
"""
struct PTU{NN <: AbstractNearestNeighbors, T<:Real} <: NonlinearDimensionalityReduction
    d::Int
    k::Real
    gauges::Array{T,3}
    model::MDS{T}
    nearestneighbors::NN
    component::AbstractVector{Int}
end

## properties
size(R::PTU) = (R.d, size(R.gauges, 2))
eigvals(R::PTU) = R.λ
neighbors(R::PTU) = R.k
vertices(R::PTU) = R.component

## show
function summary(io::IO, R::PTU)
    id, od = size(R)
    msg = isinteger(R.k) ? "neighbors" : "epsilon"
    print(io, "PTU{$(R.nearestneighbors)}(indim = $id, outdim = $od, neighbors = $(R.k))")
end

## interface functions
"""
    fit(PTU, data; k=12, maxoutdim=2, nntype=BruteForce)

Fit a local tangent space alignment model to `data`.

# Arguments
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `k`: a number of nearest neighbors for construction of local subspace representation
* `maxoutdim`: a dimension of the reduced space.
* `nntype`: a nearest neighbor construction class (derived from `AbstractNearestNeighbors`)

# Examples
```julia
M = fit(PTU, rand(3,100)) # construct PTU model
R = transform(M)           # perform dimensionality reduction
```
"""
function fit(::Type{PTU}, X::AbstractMatrix{T};
             k::Real=12, K=k, maxoutdim::Int=2, nntype=BruteForce,debug=false) where {T<:Real}
    # Construct NN graph
    d, n = size(X)
    NN = fit(nntype, X)
    E, _ = adjacency_list(NN, X, K)
    _, C = largest_component(SimpleGraph(n, E))
    A = adjacency_matrix(NN, X, k)
    G, C2 = largest_component(SimpleGraph(A))

    # Correct indexes of neighbors if more then one connected component
    fixindex = length(C) < n
    if fixindex
        n = length(C)
        R = Dict(zip(C, collect(1:n)))
    end

    # construct orthognoal basis
    Θ = zeros(T, d,maxoutdim,n)
    #Θ = Vector{SMatrix{d,maxoutdim, Float64, d*maxoutdim}}(undef, n)
    ΔTsb = 0.0
    for (ii,i) in enumerate(C)
        NI = E[i] # neighbor's indexes

        # fix indexes for connected components
        NIfix, NIcc = if fixindex # fix index
            JJ = [i for i in NI if i ∈ C] # select points that are in CC
            KK = [R[i] for i in JJ if haskey(R, i)] # convert NI to CC index
            JJ, KK
        else
            NI, NI
        end
        l = length(NIfix)
        l == 0 && continue # skip

        # re-center points in neighborhood
        VX = view(X, :, NIfix)
        δ_x = VX .- view(X, :, i:i) 

        # Compute orthogonal basis H of θ'
        ss = svd(δ_x)
        t0 = time()
        Up = standardize_basis(ss.U)
        ΔTsb += time() - t0
        Θ[:,:,ii] = Up[:,1:maxoutdim]
    end
    # compute shortest path for every point
    DD = zeros(T,n,n)
    R = diagm(ones(T, maxoutdim))
    R2 = similar(R)
    R3 = similar(R)
    θ = zeros(T, d,maxoutdim)
    if debug
        V = zeros(T, d)
    else
        V = zeros(T, maxoutdim)
    end
    # debug timing info
    ΔTdj = 0.0
    ΔTp = 0.0
    ΔT0 = 0.0
    ΔT1 = 0.0
    ΔT2 = 0.0
    ΔT3 = 0.0
    nt = 0

    # temporary variables
    Rq = fill(0.0, maxoutdim, maxoutdim, n,n)
    qq = fill(false, n,n)
    θp = fill(0.0, maxoutdim, d)
    ΔX = fill(0.0, d)
    v = fill(0.0, maxoutdim)

    for i in 1:n 
        t0 = time()
        dj = dijkstra_shortest_paths(G,i,A;trackvertices=true)
        ΔTdj += time() - t0
        # modified geodesic distance
        # TODO: This is very convoluted (and probably not quite right yet. It would better to 
        # modify the dijkstra algorithm directly)
        for k in 2:n
            # get the path from i to closest_vertices[k]
            kn = dj.closest_vertices[k]
            fill!(V,zero(T))
            R .= I(maxoutdim)
            jp = i
            θ0 = view(Θ,:,:,jp)
            t0 = time()
            path = get_path(dj, kn)
            ΔTp += time() - t0
            for l in 2:length(path)
                j1 = path[l-1]
                j2 = path[l]
                #ss = svd(θ'*Θ[:,:,j1])
                if debug
                    # no projection
                    V .+= X[:,C2[j2]]-X[:,C2[j1]]
                    DD[i,kn] += norm(X[:,C2[j2]]-X[:,C2[j1]])
                else
                    θ1 = view(Θ,:,:,j1)
                    # check if we have already computed this alignment
                    if qq[jp,j1] == false
                        t0 = time()
                        ss = svd(θ0'*θ1)
                        ΔT1 += time() - t0
                        #R .= R*(ss.V*ss.U')'
                        nt += 1
                        Rq[:,:, jp,j1] = ss.U*ss.Vt
                        qq[jp,j1] = true
                    end
                    t0 = time()
                    #mul!(R2, R, view(Rq, :,:,jp,j1))
                    #R2 .= Rq[:,:,jp,j1]
                    copy!(R2, view(Rq,:,:,jp,j1))
                    #R2 = view(Rq,:,:,jp,j1)
                    ΔT0 += time() - t0
                    t0 = time()
                    #R .= R*view(Rq,:,:, jp,j1)
                    #R .= R*R2
                    mul!(R3, R, R2)
                    copy!(R,R3)
                    ΔT2 += time() - t0
                    t0 = time()
                    copy!(ΔX, view(X, :, C2[j2]))
                    ΔX .-= view(X, :, C2[j1])
                    mul!(θp, R, θ1')
                    mul!(v, θp, ΔX)
                    #v = R*θ1'*ΔX
                    ΔT3 += time() - t0
                    V .+= v
                    θ0 = θ1
                    jp = j1
                end
            end
            #DD[i,kn] = sum(abs2, V) 
            if debug == false
                DD[i,kn] = norm(V)
            else
                #DD[i,kn] = sqrt(DD[i,kn])
            end
        end
    end
    # broadcast!(x->-x*x/2, DD, DD)
    #symmetrize!(DD) # error in MvStats
    @show ΔT0, ΔT1, ΔT2, ΔT3, ΔTdj, ΔTp, ΔTsb, nt
    if debug
        broadcast!(x->-x*x/2, DD, DD)
        DD = (DD+DD')/2
        M = fit(KernelPCA, DD, kernel=nothing, maxoutdim=maxoutdim)
        return Isomap{nntype}(d, k, M, NN, C)
    else
        DD = (DD+DD')/2
        M = fit(MDS, DD, distances=true, maxoutdim=maxoutdim)
        return PTU{nntype,T}(d, k,Θ, M, NN, C)
    end
end

"""
    predict(R::PTU)

Transforms the data fitted to the local tangent space alignment model `R` into a reduced space representation.
"""
predict(R::PTU) = predict(R.model)

