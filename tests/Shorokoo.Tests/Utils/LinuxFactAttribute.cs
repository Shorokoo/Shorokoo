namespace Shorokoo.Tests.Utils;

/// <summary>Runs a test only on Linux: one about what Linux itself does, such as how it maps a
/// process's memory.</summary>
public sealed class LinuxFactAttribute : FactAttribute
{
    public LinuxFactAttribute()
    {
        if (!OperatingSystem.IsLinux())
            Skip = "What this covers is Linux's own.";
    }
}
