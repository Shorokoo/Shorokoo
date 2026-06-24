using System;
using System.Reflection;

namespace Shorokoo
{
    /// <summary>
    /// Framework name and version, as stamped into exported ONNX models
    /// (<c>producer_name</c> / <c>producer_version</c>).
    ///
    /// The version is read from the assembly's informational version, which MinVer
    /// stamps at build time from the git tag (any "+build-metadata" suffix is
    /// stripped). This keeps the runtime version in lockstep with the published
    /// package version, with the git tag as the single source of truth.
    /// </summary>
    public static class ShorokooVersion
    {
        static readonly string s_informational =
            typeof(ShorokooVersion).Assembly
                .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? "0.0.0";

        public const string Name = "Shorokoo";

        /// <summary>Full SemVer version, e.g. "0.1.0" or "0.1.0-preview.1".</summary>
        public static string VersionString => s_informational.Split('+')[0];

        /// <summary>Numeric version with any prerelease/build labels removed.</summary>
        public static Version Version => System.Version.Parse(VersionString.Split('-')[0]);

        public static string VersionnedName => $"{Name} v{VersionString}";
    }
}
