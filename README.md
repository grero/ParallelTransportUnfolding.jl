# ParallelTransportUnfolding

Manifold learning based on parallel transport unfolding, based on Budninskiy, Max, Glorian Yin, Leman Feng, Yiying Tong, and Mathieu Desbrun. “Parallel Transport Unfolding: A Connection-Based Manifold Learning Approach.” arXiv, November 2, 2018. [http://arxiv.org/abs/1806.09039](http://arxiv.org/abs/1806.09039).


## Usage

```julia
using ManifoldLearning
using ParallelTransportUnfolding

X,segments = ManifoldLearning.swiss_roll(;segments=5);
ptu = fit(PTU, X;k=10)
Y = predict(ptu)
```

