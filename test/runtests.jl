using ParallelTransportUnfolding
using ParallelTransportUnfolding: adjacency_matrix, SimpleGraph, get_path, get_path_length, largest_component, dijkstra_shortest_paths
using Test


@testset "Geodecics" begin
    # random points on a sphere
    X = randn(3,1000) 
    X ./= sqrt.(sum(abs2,X, dims=1))

    # choose two points
    θ1 = π/4
    θ2 = π/2
    ϕ1 = π/3
    ϕ2 = 2π/3
    r = 1.0
    X[1,1] = cos(θ1)*cos(ϕ1)
    X[2,1] = sin(θ1)*cos(ϕ1)
    X[3,1] = sin(ϕ1)

    X[1,2] = cos(θ2)*cos(ϕ2)
    X[2,2] = sin(θ2)*cos(ϕ2)
    X[3,2] = sin(ϕ2)

    d0 = sqrt(sum((X[:,1]-X[:,2]).^2))
    @show d0
    
    # geodesic is just a great circle
    d = acos(sin(ϕ1)*sin(ϕ2) + cos(ϕ1)*cos(ϕ2)*cos(θ2-θ1))
    @show d
    pq = fit(PTU, X;k=10)

    dj = dijkstra_shortest_paths(pq.proximity_graph,1,pq.adjacency_matrix;trackvertices=true)
    path = get_path(dj, 2)
    dg = get_path_length(path, X, pq.gauges) 
    @show dg
    @test d0 < dg <= d
end
