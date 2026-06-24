using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using Shorokoo;
using Shorokoo.Onnx;

namespace Shorokoo.Core.Utils
{
    internal static class Extensions
    {
        public static void AddAll<T>(this ICollection<T> col, IEnumerable<T> toAdd)
        {
            foreach (var item in toAdd)
                col.Add(item);
        }

        public static void AddAll<K, V>(this IDictionary<K, V> dct, IEnumerable<(K key, V value)> toAdd)
        {
            foreach ((var k, var v) in toAdd)
                dct.Add(k, v);
        }

        public static void AddAll<K, V>(this IDictionary<K, V> dct, IEnumerable<Tuple<K, V>> toAdd)
        {
            foreach ((var k, var v) in toAdd)
                dct.Add(k, v);
        }

        public static IEnumerable<(T item, int index)> Iterate<T>(this IEnumerable<T> items)
        {
            return items.Select((item, index) => (item, index));
        }

        public static Index FindIndexOf<T>(this IEnumerable<T> source, T value)
        {
            int index = 0;
            foreach (var item in source)
            {
                if (EqualityComparer<T>.Default.Equals(item, value))
                    return index;
                index++;
            }
            return ^0; // Not found
        }

        public static IEnumerable<T> NotNulls<T>(this IEnumerable<T?> toNotNulls)
            where T : class
        {
            foreach (var item in toNotNulls)
                if (item is not null)
                    yield return item;
        }

        public static IEnumerable<T> NotNulls<T>(this IEnumerable<T?> toNotNulls)
            where T : struct
        {
            foreach (var item in toNotNulls)
                if (item is not null)
                    yield return item.Value;
        }

        public static IEnumerable<T> AssertNotNulls<T>(this IEnumerable<T?> areNotNull)
        {
#if DEBUG
            return slowAssertNotNull(areNotNull);
#else
            return (IEnumerable<T>)areNotNull;
#endif
        }

        public static IEnumerable ToEnumerable(this ITuple tuple)
        {
            for (var i = 0; i < tuple.Length; i++)
                yield return tuple[i];
        }

        public static IEnumerable<T> Cast<T>(this ITuple tuple)
         => tuple.ToEnumerable().Cast<T>();


        public static T AssertNotNull<T>(this T? checkNull) where T : class
        {
            Debug.Assert(checkNull is not null);
            return checkNull;
        }

        public static T AssertNotNull<T>(this T? checkNull) where T : struct
        {
            Debug.Assert(checkNull is not null);
            return checkNull.Value;
        }

        public static T NotNull<T>(this T? checkNull) where T : class
        {
            if (checkNull is null)
                throw new InvalidTensorOperationException(ErrorCodes.EXT002, "NotNull", typeof(T).Name, 
                    "Reference type value is null when it should not be");

            return checkNull;
        }

        public static T NotNull<T>(this T? checkNull) where T : struct
        {
            if (checkNull is null)
                throw new InvalidTensorOperationException(ErrorCodes.EXT003, "NotNull", typeof(T).Name, 
                    "Nullable struct value is null when it should not be");

            return checkNull.Value;
        }

        private static IEnumerable<T> slowAssertNotNull<T>(this IEnumerable<T?> areNotNull)
        {
            foreach (var item in areNotNull)
            {
                Debug.Assert(item != null);
                yield return item;
            }
        }

        public static IEnumerable<T> NotNull<T>(this IEnumerable<T?> toNotNulls)
            where T : struct
        {
            foreach (var item in toNotNulls)
                if (item != null)
                    yield return item.Value;
        }

        public static IEnumerable<TOut> Convert<TIn, TOut>(this IEnumerable<TIn> source)
        {
            var castMethod = typeof(TOut).GetMethod("op_Implicit", new[] { typeof(TIn) }) ?? typeof(TOut).GetMethod("op_Explicit", new[] { typeof(TIn) });
            if (castMethod is not null)
                return source.Select(x => x is null ? default! : (TOut)(castMethod.Invoke(null, new object[] { x }).AssertNotNull()));

            return source.Select(x => System.Convert.ChangeType(x, typeof(TOut))).Cast<TOut>().ToArray();
        }

        public static IEnumerable<TOut> Convert<TOut>(this IEnumerable<int> source)
        {
            switch (default(TOut))
            {
                case int:
                    return (IEnumerable<TOut>)source;
                case uint:
                    return (IEnumerable<TOut>)CastIteratorInt32UInt32(source);
                case long:
                    return (IEnumerable<TOut>)CastIteratorInt32Int64(source);
                case ulong:
                    return (IEnumerable<TOut>)CastIteratorInt32UInt64(source);
            }

            throw new UnsupportedDTypeException(ErrorCodes.EXT004, typeof(TOut).Name, "Convert<int>", 
                $"Cannot convert from int to '{typeof(TOut).FullName}'. Supported target types: int, uint, long, ulong");
        }

        public static IEnumerable<TOut> Convert<TOut>(this IEnumerable<uint> source)
        {
            switch (default(TOut))
            {
                case int:
                    return (IEnumerable<TOut>)CastIteratorUInt32Int32(source);
                case uint:
                    return (IEnumerable<TOut>)source;
                case long:
                    return (IEnumerable<TOut>)CastIteratorUInt32Int64(source);
                case ulong:
                    return (IEnumerable<TOut>)CastIteratorUInt32UInt64(source);
            }

            throw new UnsupportedDTypeException(ErrorCodes.EXT005, typeof(TOut).Name, "Convert<uint>", 
                $"Cannot convert from uint to '{typeof(TOut).FullName}'. Supported target types: int, uint, long, ulong");
        }

        public static IEnumerable<TOut> Convert<TOut>(this IEnumerable<long> source)
        {
            switch (default(TOut))
            {
                case int:
                    return (IEnumerable<TOut>)CastIteratorInt64Int32(source);
                case uint:
                    return (IEnumerable<TOut>)CastIteratorInt64UInt32(source);
                case long:
                    return (IEnumerable<TOut>)source;
                case ulong:
                    return (IEnumerable<TOut>)CastIteratorInt64UInt64(source);
            }

            throw new UnsupportedDTypeException(ErrorCodes.EXT006, typeof(TOut).Name, "Convert<long>", 
                $"Cannot convert from long to '{typeof(TOut).FullName}'. Supported target types: int, uint, long, ulong");
        }

        public static IEnumerable<TOut> Convert<TOut>(this IEnumerable<ulong> source)
        {
            switch (default(TOut))
            {
                case int:
                    return (IEnumerable<TOut>)CastIteratorUInt64Int32(source);
                case uint:
                    return (IEnumerable<TOut>)CastIteratorUInt64UInt32(source);
                case long:
                    return (IEnumerable<TOut>)CastIteratorUInt64Int64(source);
                case ulong:
                    return (IEnumerable<TOut>)source;
            }

            throw new UnsupportedDTypeException(ErrorCodes.EXT007, typeof(TOut).Name, "Convert<ulong>", 
                $"Cannot convert from ulong to '{typeof(TOut).FullName}'. Supported target types: int, uint, long, ulong");
        }


        private static IEnumerable<int> CastIteratorUInt32Int32(IEnumerable<uint> source)
        { foreach (long obj in source) yield return (int)obj; }

        private static IEnumerable<int> CastIteratorInt64Int32(IEnumerable<long> source)
        { foreach (long obj in source) yield return (int)obj; }

        private static IEnumerable<int> CastIteratorUInt64Int32(IEnumerable<ulong> source)
        { foreach (long obj in source) yield return (int)obj; }


        private static IEnumerable<uint> CastIteratorInt32UInt32(IEnumerable<int> source)
        { foreach (long obj in source) yield return (uint)obj; }

        private static IEnumerable<uint> CastIteratorInt64UInt32(IEnumerable<long> source)
        { foreach (long obj in source) yield return (uint)obj; }

        private static IEnumerable<uint> CastIteratorUInt64UInt32(IEnumerable<ulong> source)
        { foreach (long obj in source) yield return (uint)obj; }


        private static IEnumerable<long> CastIteratorInt32Int64(IEnumerable<int> source)
        { foreach (long obj in source) yield return (long)obj; }

        private static IEnumerable<long> CastIteratorUInt32Int64(IEnumerable<uint> source)
        { foreach (long obj in source) yield return (long)obj; }

        private static IEnumerable<long> CastIteratorUInt64Int64(IEnumerable<ulong> source)
        { foreach (long obj in source) yield return (long)obj; }


        private static IEnumerable<ulong> CastIteratorInt32UInt64(IEnumerable<int> source)
        { foreach (long obj in source) yield return (ulong)obj; }

        private static IEnumerable<ulong> CastIteratorUInt32UInt64(IEnumerable<uint> source)
        { foreach (long obj in source) yield return (ulong)obj; }

        private static IEnumerable<ulong> CastIteratorInt64UInt64(IEnumerable<long> source)
        { foreach (long obj in source) yield return (ulong)obj; }
    }
}
