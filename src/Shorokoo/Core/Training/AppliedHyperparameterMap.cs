using System;
using System.Collections;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;

namespace Shorokoo
{
    /// <summary>
    /// The immutable map behind <see cref="TrainingCheckpoint.AppliedHyperparameters"/> and
    /// <see cref="TrainingHistoryEntry.AppliedHyperparameters"/>: one value per hyperparameter name,
    /// enumerated in the order the names were given — a step's map in its rig's
    /// <see cref="TrainingRig.HyperparameterNames"/> order, a copy of another dictionary in that one's
    /// enumeration order, and a map read back from a saved history in the order the history first
    /// holds each name (<see cref="TrainingHistory.HyperparameterNames"/>). It has no mutating surface
    /// and nothing it exposes can be cast back to a mutable collection.
    ///
    /// <para>The names and their lookup index are a <see cref="Layout"/>, built once and shared by
    /// every map over the same names — every step a rig runs uses the rig's one layout — so a step's
    /// map costs one array of values.</para>
    /// </summary>
    internal sealed class AppliedHyperparameterMap : IReadOnlyDictionary<string, AppliedHyperparameter>
    {
        /// <summary>An ordered set of hyperparameter names and the index that finds each.</summary>
        internal sealed class Layout
        {
            private Layout(ImmutableArray<string> names, FrozenDictionary<string, int> index)
            {
                Names = names;
                Index = index;
            }

            internal ImmutableArray<string> Names { get; }

            internal FrozenDictionary<string, int> Index { get; }

            /// <summary>A layout over <paramref name="names"/>, in that order.</summary>
            /// <exception cref="ArgumentException">A name repeats.</exception>
            internal static Layout Of(IEnumerable<string> names)
            {
                var ordered = ImmutableArray.CreateRange(names);
                var index = new Dictionary<string, int>(ordered.Length, StringComparer.Ordinal);
                for (int i = 0; i < ordered.Length; i++)
                    if (!index.TryAdd(ordered[i], i))
                        throw new ArgumentException($"Hyperparameter '{ordered[i]}' is named twice.", nameof(names));
                return new Layout(ordered, index.ToFrozenDictionary(StringComparer.Ordinal));
            }
        }

        private readonly Layout _layout;
        private readonly ImmutableArray<AppliedHyperparameter> _values;

        /// <summary>A map giving <paramref name="values"/>[i] to <paramref name="layout"/>'s i-th name.</summary>
        internal AppliedHyperparameterMap(Layout layout, ImmutableArray<AppliedHyperparameter> values)
        {
            if (values.Length != layout.Names.Length)
                throw new ArgumentException(
                    $"{values.Length} value(s) for {layout.Names.Length} hyperparameter name(s).", nameof(values));
            foreach (var v in values)
                if (v is null) throw new ArgumentException("A hyperparameter value is null.", nameof(values));
            _layout = layout;
            _values = values;
        }

        /// <summary><paramref name="source"/> itself where it is already such a map, else an immutable
        /// copy of it in its enumeration order.</summary>
        internal static AppliedHyperparameterMap Of(IReadOnlyDictionary<string, AppliedHyperparameter> source)
        {
            if (source is AppliedHyperparameterMap map) return map;
            var names = new List<string>(source.Count);
            var values = ImmutableArray.CreateBuilder<AppliedHyperparameter>(source.Count);
            foreach (var (name, value) in source)
            {
                names.Add(name);
                values.Add(value ?? throw new ArgumentException($"Hyperparameter '{name}' has a null value.", nameof(source)));
            }
            return new AppliedHyperparameterMap(Layout.Of(names), values.MoveToImmutable());
        }

        /// <summary>Whether both maps hold the same names, each with an equal value.</summary>
        internal static bool ValueEquals(
            IReadOnlyDictionary<string, AppliedHyperparameter> a, IReadOnlyDictionary<string, AppliedHyperparameter> b)
        {
            if (ReferenceEquals(a, b)) return true;
            if (a.Count != b.Count) return false;
            foreach (var (name, value) in a)
                if (!b.TryGetValue(name, out var other) || !value.Equals(other)) return false;
            return true;
        }

        /// <inheritdoc />
        public AppliedHyperparameter this[string key]
            => _layout.Index.TryGetValue(key, out var i)
                ? _values[i]
                : throw new KeyNotFoundException($"No hyperparameter is named '{key}'.");

        /// <inheritdoc />
        public IEnumerable<string> Keys => _layout.Names;

        /// <inheritdoc />
        public IEnumerable<AppliedHyperparameter> Values => _values;

        /// <inheritdoc />
        public int Count => _values.Length;

        /// <inheritdoc />
        public bool ContainsKey(string key) => _layout.Index.ContainsKey(key);

        /// <inheritdoc />
        public bool TryGetValue(string key, [MaybeNullWhen(false)] out AppliedHyperparameter value)
        {
            if (_layout.Index.TryGetValue(key, out var i))
            {
                value = _values[i];
                return true;
            }
            value = null;
            return false;
        }

        /// <inheritdoc />
        public IEnumerator<KeyValuePair<string, AppliedHyperparameter>> GetEnumerator()
        {
            for (int i = 0; i < _values.Length; i++)
                yield return new KeyValuePair<string, AppliedHyperparameter>(_layout.Names[i], _values[i]);
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
