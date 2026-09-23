using Shorokoo.Core.Factory.IR;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Shorokoo.Core;
using Shorokoo.Core.Training;

namespace Shorokoo
{
    public class TensorDataModelParam : NamedModelParam
    {
        private TensorData data;

        public Shape ParamShape { get; protected set; } = null!;

        public TensorDataModelParam(string name, ModelParamType paramType, TensorData data) : base()
        {
            this.data = data;
            this.ParamShape = data.Shape;
            base.ParamName = name;
            base.Type = data.DType;
            base.ParamType = paramType;
        }

        public override IShorokooTensorValue ToTensorValue()
        {
            return data.ToTensorValue();
        }

        /// <inheritdoc/>
        internal override IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
            => data.ToTensorValue(backend);

        public override TensorData ToTensorData() => data;
        public override TensorData<T> ToTensorData<T>() => (TensorData<T>)data;

        public override TensorDataSequence ToTensorDataSequence()
        {
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorDataSequence", "TensorDataModelParam",
                "TensorDataModelParam does not support ToTensorDataSequence. Use ToTensorData() instead");
        }

        public override TensorDataSequence<T> ToTensorDataSequence<T>()
        {
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorDataSequence<T>", "TensorDataModelParam",
                "TensorDataModelParam does not support ToTensorDataSequence. Use ToTensorData<T>() instead");
        }

        public override string ToString()
        {
            return "{" + this.ParamName + ":" + this.ParamShape + ":" + this.Type + "}";
        }

    }

    /// <summary>
    /// A feed the caller has given away rather than lent — <see cref="TensorData.Donate"/>'s, as a
    /// model parameter. It feeds the same value <see cref="TensorDataModelParam"/> does, and differs
    /// in what the run does with the tensor: a run consumes it when it starts, rather than taking a
    /// reader lock on it, and releases its memory as soon as it returns
    /// (Shorokoo/Shorokoo#359).
    /// </summary>
    public sealed class DonatedTensorModelParam : TensorDataModelParam
    {
        /// <summary>Wraps <paramref name="donation"/>'s tensor as the parameter named
        /// <paramref name="name"/>.</summary>
        /// <exception cref="ArgumentNullException"><paramref name="donation"/> is null.</exception>
        public DonatedTensorModelParam(string name, ModelParamType paramType, TensorDonation donation)
            : base(name, paramType, (donation ?? throw new ArgumentNullException(nameof(donation))).Tensor)
        {
        }
    }

    public class TensorDataSequenceModelParam : NamedModelParam
    {
        private TensorDataSequence data;

        public int Count { get; protected set; }

        public TensorDataSequenceModelParam(string name, ModelParamType paramType, TensorDataSequence data) : base()
        {
            this.data = data;
            this.Count = data.Count;
            base.ParamName = name;
            base.Type = data.DType;
            base.ParamType = paramType;
            base.Structure = DataStructure.Sequence;
        }

        public override IShorokooTensorValue ToTensorValue() => data.ToTensorValue();

        /// <inheritdoc/>
        internal override IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
            => data.ToTensorValue(backend);

        public override TensorData ToTensorData() =>
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorData", "TensorDataSequenceModelParam",
                "TensorDataSequenceModelParam does not support ToTensorData. Use ToTensorDataSequence() instead");

        public override TensorData<T> ToTensorData<T>() =>
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorData<T>", "TensorDataSequenceModelParam",
                "TensorDataSequenceModelParam does not support ToTensorData. Use ToTensorDataSequence<T>() instead");


        public override TensorDataSequence ToTensorDataSequence() => data;

        public override TensorDataSequence<T> ToTensorDataSequence<T>() => data.As<T>();

        public override string ToString()
        {
            return "{" + this.ParamName + ":" + this.Count + ":" + this.Type + "}";
        }

    }

    /// <summary>
    /// Model parameter that holds an <see cref="OptionalTensorData"/> — the value of an
    /// OptionalTensor input/output (present or absent).
    /// </summary>
    public class OptionalTensorDataModelParam : NamedModelParam
    {
        public OptionalTensorData Data { get; }

        public OptionalTensorDataModelParam(string name, ModelParamType paramType, OptionalTensorData data) : base()
        {
            this.Data = data;
            base.ParamName = name;
            base.Type = data.DType;
            base.ParamType = paramType;
            base.Structure = DataStructure.Optional;
        }

        public override IShorokooTensorValue ToTensorValue() => ToTensorValue(DefaultBackend.Instance);

        /// <inheritdoc/>
        internal override IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
        {
            // ONNX Runtime accepts a plain tensor where an optional input is expected (opset 18+),
            // so a present optional feeds its inner tensor directly.
            if (Data.HasValue && Data.Value is not null)
                return Data.Value.ToTensorValue(backend);
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorValue", "OptionalTensorDataModelParam",
                "An absent OptionalTensorData cannot be fed to the ONNX Runtime session as a none-optional input. " +
                "Execute through the QuickExecutionEngine (QuickExecutionEngine.Execute), which supports absent optionals.");
        }

        public override TensorData ToTensorData() => Data.Value
            ?? throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorData", "OptionalTensorDataModelParam",
                "The optional is absent; there is no tensor value to return. Check HasValue first.");

        public override TensorData<T> ToTensorData<T>() => (TensorData<T>)ToTensorData();

        public OptionalTensorData ToOptionalTensorData() => Data;

        public override TensorDataSequence ToTensorDataSequence() =>
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorDataSequence", "OptionalTensorDataModelParam",
                "OptionalTensorDataModelParam is not a sequence. Use ToOptionalTensorData().");

        public override TensorDataSequence<T> ToTensorDataSequence<T>() =>
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorDataSequence<T>", "OptionalTensorDataModelParam",
                "OptionalTensorDataModelParam is not a sequence. Use ToOptionalTensorData().");

        public override string ToString() => "{" + this.ParamName + ":" + this.Data + "}";
    }

    /// <summary>
    /// Model parameter that holds TensorStruct data.
    /// Used when a TensorStruct is a model input/output.
    /// </summary>
    public class TensorStructModelParam : NamedModelParam
    {
        private TensorDataStruct data;

        public TensorDataStruct StructData => data;

        public TensorStructDef Definition => data.Definition;

        public TensorStructModelParam(string name, ModelParamType paramType, TensorDataStruct data) : base()
        {
            this.data = data;
            base.ParamName = name;
            base.Type = DType.GetOrCreateForTensorStruct(data.Definition);
            base.ParamType = paramType;
            base.Structure = DataStructure.TensorStruct;
        }

        public override IShorokooTensorValue ToTensorValue()
        {
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorValue", "TensorStructModelParam",
                "TensorStructModelParam cannot be directly converted to a tensor value. Use individual field data via StructData.GetField()");
        }

        public override TensorData ToTensorData()
        {
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorData", "TensorStructModelParam",
                "TensorStructModelParam does not support ToTensorData. Use StructData.GetField() to access individual field data");
        }

        public override TensorData<T> ToTensorData<T>()
        {
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorData<T>", "TensorStructModelParam",
                "TensorStructModelParam does not support ToTensorData. Use StructData.GetField() to access individual field data");
        }

        public override TensorDataSequence ToTensorDataSequence()
        {
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorDataSequence", "TensorStructModelParam",
                "TensorStructModelParam does not support ToTensorDataSequence. Use StructData.GetField() to access individual field data");
        }

        public override TensorDataSequence<T> ToTensorDataSequence<T>()
        {
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorDataSequence<T>", "TensorStructModelParam",
                "TensorStructModelParam does not support ToTensorDataSequence. Use StructData.GetField() to access individual field data");
        }

        public IData GetFieldData(string fieldName) => data.Fields[fieldName];

        public override string ToString()
        {
            var fieldNames = string.Join(",", data.Fields.Keys);
            return "{" + this.ParamName + ":struct[" + fieldNames + "]:" + this.Type + "}";
        }
    }


    public abstract class NamedModelParam
    {
        public string ParamName { get; protected set; } = null!;

        public ModelParamType ParamType { get; protected set; }

        public long NumBytes { get; protected set; }

        public long BitsPerElement { get; protected set; }

        public DType Type { get; protected set; } = null!;

        public DataStructure Structure { get; protected set; } = DataStructure.Tensor;

        public abstract IShorokooTensorValue ToTensorValue();

        /// <summary>
        /// This parameter as a value of <paramref name="backend"/>'s runtime — the form the feed
        /// sites take, so that a literal is materialised by the backend whose session is about to
        /// read it rather than by whichever one the process happens to default to. A value belongs
        /// to the runtime that made it, and feeding one built elsewhere costs a rebuild per run at
        /// best (<c>OrtSession.Unwrap</c>) and is not possible at all for data that has
        /// left host memory.
        ///
        /// <para>The default answers the same as the no-argument form, which is right for a
        /// parameter with no contents of its own to place — a <see cref="TensorStructModelParam"/>
        /// refuses either way — and keeps any parameter type written before backends could differ
        /// working exactly as it did.</para>
        ///
        /// <para>The value returned is the parameter's own: read it, do not dispose it.</para>
        /// </summary>
        internal virtual IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
            => ToTensorValue();

        public abstract TensorData ToTensorData();

        public abstract TensorData<T> ToTensorData<T>() where T : IVarType;

        public abstract TensorDataSequence ToTensorDataSequence();
        public abstract TensorDataSequence<T> ToTensorDataSequence<T>() where T : IVarType;

        public static NamedModelParam FromIData(string name, ModelParamType paramType, IData data)
        {
            if (data is TensorData td)
                return new TensorDataModelParam(name, paramType, td);
            else if (data is TensorDonation donation)
                return new DonatedTensorModelParam(name, paramType, donation);
            else if (data is OptionalTensorData otd)
                return new OptionalTensorDataModelParam(name, paramType, otd);
            else if (data is TensorDataSequence tds)
                return new TensorDataSequenceModelParam(name, paramType, tds);
            else if (data is TensorDataStruct tdsStruct)
                return new TensorStructModelParam(name, paramType, tdsStruct);
            else
                throw new InvalidTensorOperationException(ErrorCodes.FW008, data.GetType().Name, "FromIData",
                    "Unsupported IData type for NamedModelParam: " + data.GetType().Name);
        }
    }
}
