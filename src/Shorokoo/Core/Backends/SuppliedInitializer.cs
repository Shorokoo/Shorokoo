namespace Shorokoo.Core.Backends;

/// <summary>
/// One of a model's initializers handed to its session as a value already in the backend's memory
/// rather than carried in the model's bytes: a weight loaded straight onto the card
/// (Shorokoo/Shorokoo#436). The session reads <see cref="Value"/> where it is, for as long as it
/// lives.
///
/// <para>The model declares such an initializer by its <see cref="Name"/>, with its element type
/// and shape, as external data at <see cref="PlaceholderLocation"/> — which holds nothing, the
/// value arriving from here instead — and lists it among the graph's inputs too. The second makes
/// it overridable, which keeps a runtime from folding a node that reads it into a constant at
/// build: a fold would read the value from the placeholder rather than from here.</para>
/// </summary>
/// <param name="Name">The initializer's name in the model.</param>
/// <param name="Value">The value, in the memory of the backend the session is built on.</param>
public sealed record SuppliedInitializer(string Name, IShorokooTensorValue Value)
{
    /// <summary>The external-data location a supplied initializer is declared at.</summary>
    public const string PlaceholderLocation = "shorokoo-supplied-initializer";
}
