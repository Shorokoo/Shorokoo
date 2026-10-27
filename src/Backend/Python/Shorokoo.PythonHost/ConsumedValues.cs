using Shorokoo.Core.Backends;

namespace Shorokoo.PythonHost;

/// <summary>
/// The values a run was handed over (<see cref="IShorokooSession.RunConsuming(IReadOnlyDictionary{string, IShorokooTensorValue}, IReadOnlyCollection{IShorokooTensorValue}, IReadOnlyList{string}, RunSettings)"/>),
/// which the session releases, each exactly once, however the run ends.
/// </summary>
internal static class ConsumedValues
{
    /// <summary>Releases every value of <paramref name="consumed"/> through
    /// <paramref name="release"/>: one whose release throws stops none of the others. Returns what
    /// failed — the one failure, or every one together — or null where none did.</summary>
    public static Exception? ReleaseEach(IEnumerable<IShorokooTensorValue> consumed, Action<IShorokooTensorValue> release)
    {
        List<Exception>? failures = null;
        foreach (var value in consumed)
        {
            try
            {
                release(value);
            }
            catch (Exception failure) when (failure is not OutOfMemoryException)
            {
                (failures ??= []).Add(failure);
            }
        }
        return failures switch
        {
            null => null,
            [var only] => only,
            _ => new AggregateException(failures),
        };
    }
}
