using System.Reflection;

namespace Shorokoo.Tests.Utils;

/// <summary>
/// Runs a static method of this assembly in a process of its own, through
/// <see cref="SideBySideBackendHardwareTests.InAChildProcess"/>: for a test whose regression is a
/// read past the memory it was handed or a stack overflow, which ends the process rather than
/// failing an assertion, so that it fails the one test rather than the whole run.
/// </summary>
internal static class OwnProcess
{
    internal const string Case = "own-process";

    internal static void Run(Type type, string method)
        => Assert.Equal(0, SideBySideBackendHardwareTests.InAChildProcess([], Case, type.FullName!, method));

    internal static int Child(string type, string method)
    {
        try
        {
            typeof(OwnProcess).Assembly.GetType(type, throwOnError: true)!
                .GetMethod(method, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)!
                .Invoke(null, null);
            return 0;
        }
        catch (Exception)
        {
            return 1;
        }
    }
}
