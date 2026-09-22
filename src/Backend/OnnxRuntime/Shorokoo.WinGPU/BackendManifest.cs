using Shorokoo.Core.Backends;

// What this backend is and what it needs, readable out of the file's metadata without loading it,
// so BackendPackage.Probe can turn it away on the wrong machine before anything native is touched.
[assembly: ShorokooBackend("windows", "x64", "cuda",
    Natives = "onnxruntime.dll;onnxruntime_providers_cuda.dll;onnxruntime_providers_shared.dll",
    RequiresCudaRuntime = "12")]
