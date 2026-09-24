using Shorokoo.Core.Backends;

// What this backend is and what it needs, readable out of the file's metadata without loading it.
// It ships no native library of its own and declares no CUDA runtime: PyTorch and the CUDA
// libraries it loads come from its Python environment, so the machine needs only the driver, which
// starting the backend checks. It is never a discovery candidate: it runs where a program names it.
[assembly: ShorokooBackend("linux", "x64", "cuda", Selection = ShorokooBackendAttribute.ExplicitSelection)]
