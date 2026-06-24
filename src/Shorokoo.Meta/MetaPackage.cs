// The 'Shorokoo' meta-package ships no assembly of its own
// (IncludeBuildOutput=false); it exists only to aggregate its package
// dependencies (Shorokoo.Core + Shorokoo.CodeGen). This marker keeps the
// project compiling.
namespace Shorokoo.Meta;

internal static class MetaPackage
{
}
