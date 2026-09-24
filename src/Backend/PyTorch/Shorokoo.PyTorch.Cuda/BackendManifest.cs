using Shorokoo.Core.Backends;

// What this backend is and what it needs, readable out of the file's metadata without loading it.
// It ships no native library of its own and declares no CUDA runtime: PyTorch and the CUDA 13
// libraries it loads come from its Python environment, which has a lock for each operating system
// named here, so the machine needs only an NVIDIA driver recent enough for CUDA 13 -- which a probe
// checks, and which starting the backend checks before provisioning anything. It is never a
// discovery candidate: it runs where a program names it.
[assembly: ShorokooBackend("linux;windows", "x64", "cuda",
    Selection = ShorokooBackendAttribute.ExplicitSelection,
    RequiresCudaDriver = "13.0")]
