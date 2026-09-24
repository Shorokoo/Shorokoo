using Shorokoo.Core.Backends;

// What this backend is and what it needs, readable out of the file's metadata without loading it.
// It ships no native library of its own -- PyTorch comes from its Python environment -- and it is
// never a discovery candidate: it runs where a program names it.
[assembly: ShorokooBackend("linux", "x64", "cpu", Selection = ShorokooBackendAttribute.ExplicitSelection)]
