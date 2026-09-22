using Shorokoo.Core.Backends;

// What this backend is and what it needs, readable out of the file's metadata without loading it,
// so BackendPackage.Probe can turn it away on the wrong machine before anything native is touched.
[assembly: ShorokooBackend("linux", "x64", "cpu", Natives = "libonnxruntime.so")]
