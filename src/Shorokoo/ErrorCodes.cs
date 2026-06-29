namespace Shorokoo
{
    /// <summary>
    /// Centralized constants for all Shorokoo error codes to ensure uniqueness and consistency
    /// </summary>
    public static class ErrorCodes
    {
        #region DType Error Codes (DT001-DT025)
        
        /// <summary>Int4 precision is not supported for variable type conversion</summary>
        public const string DT001 = "DT001";
        
        /// <summary>UInt4 precision is not supported for variable type conversion</summary>
        public const string DT002 = "DT002";
        
        /// <summary>Complex64 numbers are not supported for variable type conversion</summary>
        public const string DT007 = "DT007";
        
        /// <summary>Complex128 numbers are not supported for variable type conversion</summary>
        public const string DT008 = "DT008";
        
        /// <summary>Unknown DType for ToIVarType operation</summary>
        public const string DT009 = "DT009";
        
        /// <summary>Int4 precision is not supported for primitive type conversion</summary>
        public const string DT010 = "DT010";
        
        /// <summary>UInt4 precision is not supported for primitive type conversion</summary>
        public const string DT011 = "DT011";
        
        /// <summary>Complex64 numbers are not supported for primitive type conversion</summary>
        public const string DT013 = "DT013";
        
        /// <summary>Complex128 numbers are not supported for primitive type conversion</summary>
        public const string DT014 = "DT014";
        
        /// <summary>Module type cannot be converted to a primitive type</summary>
        public const string DT015 = "DT015";
        
        /// <summary>Model type cannot be converted to a primitive type</summary>
        public const string DT016 = "DT016";
        
        /// <summary>Unknown DType for ToPrimitiveType operation</summary>
        public const string DT017 = "DT017";
        
        /// <summary>Int4 precision is not supported for bit count encoding</summary>
        public const string DT018 = "DT018";
        
        /// <summary>UInt4 precision is not supported for bit count encoding</summary>
        public const string DT019 = "DT019";
        
        /// <summary>String DType has variable bit count and is not supported for bit count encoding</summary>
        public const string DT020 = "DT020";
        
        /// <summary>Complex64 numbers are not supported for bit count encoding</summary>
        public const string DT021 = "DT021";
        
        /// <summary>Complex128 numbers are not supported for bit count encoding</summary>
        public const string DT022 = "DT022";
        
        /// <summary>Module type has no fixed bit count encoding</summary>
        public const string DT023 = "DT023";
        
        /// <summary>Model type has no fixed bit count encoding</summary>
        public const string DT024 = "DT024";
        
        /// <summary>Unknown DType for EncodingBitCount operation</summary>
        public const string DT025 = "DT025";
        
        #endregion

        #region Vector Error Codes (VT001-VT010)
        
        /// <summary>Int4 precision is not supported for unit vector creation</summary>
        public const string VT001 = "VT001";
        
        /// <summary>UInt4 precision is not supported for unit vector creation</summary>
        public const string VT002 = "VT002";
        
        /// <summary>String DType is not supported for unit vector creation</summary>
        public const string VT003 = "VT003";
        
        /// <summary>Complex64 numbers are not supported for unit vector creation</summary>
        public const string VT004 = "VT004";
        
        /// <summary>Complex128 numbers are not supported for unit vector creation</summary>
        public const string VT005 = "VT005";
        
        /// <summary>Int4 precision is not supported for empty vector creation</summary>
        public const string VT006 = "VT006";
        
        /// <summary>UInt4 precision is not supported for empty vector creation</summary>
        public const string VT007 = "VT007";
        
        /// <summary>String DType is not supported for empty vector creation</summary>
        public const string VT008 = "VT008";
        
        /// <summary>Complex64 numbers are not supported for empty vector creation</summary>
        public const string VT009 = "VT009";
        
        /// <summary>Complex128 numbers are not supported for empty vector creation</summary>
        public const string VT010 = "VT010";
        
        #endregion

        #region Global Constructors Error Codes (GC001-GC005)
        
        /// <summary>Type not supported for scalar construction</summary>
        public const string GC001 = "GC001";
        
        /// <summary>Generic type not supported for scalar construction</summary>
        public const string GC002 = "GC002";
        
        /// <summary>Type not supported for variable creation</summary>
        public const string GC003 = "GC003";
        
        /// <summary>Input data array cannot be null for encoding</summary>
        public const string GC004 = "GC004";
        
        /// <summary>Insufficient bytes for decoding operation</summary>
        public const string GC005 = "GC005";
        
        #endregion

        #region OnnxUtils Error Codes (OU001-OU004)
        
        /// <summary>No suitable method found during reflection</summary>
        public const string OU001 = "OU001";
        
        /// <summary>Unsupported OrtValue type for ONNX data creation</summary>
        public const string OU002 = "OU002";
        
        /// <summary>OrtValue creation failed during model operations</summary>
        public const string OU003 = "OU003";
        
        /// <summary>Generic type not supported for DType conversion</summary>
        public const string OU004 = "OU004";
        
        #endregion

        #region Extensions Error Codes (EXT002-EXT007)

        /// <summary>Array argument is null when not null expected</summary>
        public const string EXT002 = "EXT002";
        
        /// <summary>Object argument is null when not null expected</summary>
        public const string EXT003 = "EXT003";
        
        /// <summary>Type conversion not supported for int operations</summary>
        public const string EXT004 = "EXT004";
        
        /// <summary>Type conversion not supported for uint operations</summary>
        public const string EXT005 = "EXT005";
        
        /// <summary>Type conversion not supported for long operations</summary>
        public const string EXT006 = "EXT006";
        
        /// <summary>Type conversion not supported for ulong operations</summary>
        public const string EXT007 = "EXT007";
        
        #endregion

        #region Node Error Codes (NOD001-NOD002)
        
        /// <summary>ONNX node failed with inner exception</summary>
        public const string NOD001 = "NOD001";
        
        /// <summary>Output structure type not supported for node output creation</summary>
        public const string NOD002 = "NOD002";
        
        #endregion

        #region TensorIndexerHelpers Error Codes (TIH001-TIH010)
        
        /// <summary>Index from end is out of bounds for tensor indexing</summary>
        public const string TIH001 = "TIH001";
        
        /// <summary>Mixed indexing operations not supported (slice and single index)</summary>
        public const string TIH002 = "TIH002";
        
        /// <summary>Step slices are not supported</summary>
        public const string TIH003 = "TIH003";
        
        /// <summary>Multiple OnnxObj long indexer parameters not allowed</summary>
        public const string TIH004 = "TIH004";
        
        /// <summary>Manual IndexerTensorParam construction not allowed</summary>
        public const string TIH005 = "TIH005";
        
        /// <summary>OnnxObj long parameter position constraint violation</summary>
        public const string TIH006 = "TIH006";
        
        /// <summary>Tensor indexer operation not implemented</summary>
        public const string TIH007 = "TIH007";
        
        #endregion

        #region OnnxUtils Extended Error Codes (OU005-OU010)
        
        /// <summary>DType not supported for default data array generation</summary>
        public const string OU005 = "OU005";

        #endregion

        #region NodeBuilder Error Codes (NB001-NB050)
        
        /// <summary>AttributeType validation failed for Bool type</summary>
        public const string NB001 = "NB001";
        
        /// <summary>AttributeType validation failed for Bools type</summary>
        public const string NB002 = "NB002";
        
        /// <summary>AttributeType validation failed for Long type</summary>
        public const string NB003 = "NB003";
        
        /// <summary>AttributeType validation failed for Longs type</summary>
        public const string NB004 = "NB004";
        
        /// <summary>AttributeType validation failed for Float type</summary>
        public const string NB005 = "NB005";
        
        /// <summary>AttributeType validation failed for Floats type</summary>
        public const string NB006 = "NB006";
        
        /// <summary>AttributeType validation failed for String type</summary>
        public const string NB007 = "NB007";
        
        /// <summary>AttributeType validation failed for Strings type</summary>
        public const string NB008 = "NB008";
        
        /// <summary>AttributeType validation failed for Tensor type</summary>
        public const string NB009 = "NB009";
        
        /// <summary>AttributeType validation failed for DType type</summary>
        public const string NB010 = "NB010";
        
        /// <summary>AttributeType validation failed for DTypes type</summary>
        public const string NB011 = "NB011";
        
        /// <summary>AttributeType validation failed for TypeProto type</summary>
        public const string NB012 = "NB012";
        
        /// <summary>AttributeType validation failed for unknown type</summary>
        public const string NB013 = "NB013";
        
        /// <summary>NodeBuilder operation not implemented</summary>
        public const string NB014 = "NB014";
        
        /// <summary>Invalid operation in NodeBuilder</summary>
        public const string NB015 = "NB015";
        
        /// <summary>Unknown attribute type conversion in NodeBuilder</summary>
        public const string NB016 = "NB016";
        
        /// <summary>Operation not implemented in NodeBuilder</summary>
        public const string NB017 = "NB017";
        
        /// <summary>Variadic count inference failed in NodeBuilder</summary>
        public const string NB018 = "NB018";
        
        /// <summary>Invalid index calculation in NodeBuilder</summary>
        public const string NB019 = "NB019";
        
        /// <summary>Attribute access failed - Long type expected</summary>
        public const string NB020 = "NB020";
        
        /// <summary>Invalid count value - exceeds int.MaxValue</summary>
        public const string NB021 = "NB021";
        
        /// <summary>InputDef to Input conversion failed</summary>
        public const string NB022 = "NB022";
        
        /// <summary>Invalid input sequence access</summary>
        public const string NB023 = "NB023";
        
        /// <summary>Unsupported SequenceDef type in NodeBuilder</summary>
        public const string NB024 = "NB024";
        
        /// <summary>Output structure creation failed</summary>
        public const string NB025 = "NB025";
        
        /// <summary>OutputDef to OutputInfo conversion failed</summary>
        public const string NB026 = "NB026";
        
        /// <summary>Variable type inference failed</summary>
        public const string NB027 = "NB027";
        
        /// <summary>OpName mismatch in NodeBuilder</summary>
        public const string NB028 = "NB028";
        
        /// <summary>Input validation failed in NodeBuilder</summary>
        public const string NB029 = "NB029";
        
        /// <summary>Attribute enumeration failed</summary>
        public const string NB030 = "NB030";
        
        /// <summary>Variable access failed in IndexOf operation</summary>
        public const string NB031 = "NB031";
        
        /// <summary>Output structure type incompatible with requested type</summary>
        public const string NB032 = "NB032";
        
        /// <summary>NodeBuilder operation validation failed</summary>
        public const string NB033 = "NB033";
        
        /// <summary>Node definition constraint violation</summary>
        public const string NB034 = "NB034";
        
        /// <summary>Input type conversion failed</summary>
        public const string NB035 = "NB035";
        
        /// <summary>Output structure constraint violation</summary>
        public const string NB036 = "NB036";
        
        /// <summary>Variable construction failed</summary>
        public const string NB037 = "NB037";
        
        /// <summary>Node operation constraint validation failed</summary>
        public const string NB038 = "NB038";
        
        /// <summary>Input constraint verification failed</summary>
        public const string NB039 = "NB039";
        
        /// <summary>Output constraint verification failed</summary>
        public const string NB040 = "NB040";
        
        /// <summary>Variable type constraint violation</summary>
        public const string NB041 = "NB041";
        
        /// <summary>Attribute constraint verification failed</summary>
        public const string NB042 = "NB042";
        
        /// <summary>Node definition building failed</summary>
        public const string NB043 = "NB043";
        
        /// <summary>Input sequence constraint violation</summary>
        public const string NB044 = "NB044";
        
        /// <summary>Output sequence constraint violation</summary>
        public const string NB045 = "NB045";
        
        /// <summary>Variable access constraint violation</summary>
        public const string NB046 = "NB046";
        
        /// <summary>Node constraint validation failed</summary>
        public const string NB047 = "NB047";
        
        /// <summary>Variable type verification failed</summary>
        public const string NB048 = "NB048";
        
        /// <summary>Node operation execution constraint failed</summary>
        public const string NB049 = "NB049";
        
        /// <summary>Variable constraint verification failed</summary>
        public const string NB050 = "NB050";
        
        #endregion

        #region VirtualGraph Error Codes (VG001-VG050)
        
        /// <summary>Iteration indices count mismatch</summary>
        public const string VG001 = "VG001";
        
        /// <summary>ModelId must be an iteration model id</summary>
        public const string VG002 = "VG002";
        
        /// <summary>Invalid tensor in graph inputs</summary>
        public const string VG003 = "VG003";
        
        /// <summary>Invalid tensor in graph outputs</summary>
        public const string VG004 = "VG004";
        
        /// <summary>Missing canonical tensor in inputs</summary>
        public const string VG005 = "VG005";
        
        /// <summary>Missing canonical tensor in outputs</summary>
        public const string VG006 = "VG006";
        
        /// <summary>Input length mismatch in graph operation</summary>
        public const string VG007 = "VG007";
        
        /// <summary>Output length mismatch in graph operation</summary>
        public const string VG008 = "VG008";
        
        /// <summary>Duplicate canonical inputs detected</summary>
        public const string VG009 = "VG009";
        
        /// <summary>Duplicate canonical outputs detected</summary>
        public const string VG010 = "VG010";
        
        /// <summary>Duplicate outputs not supported</summary>
        public const string VG011 = "VG011";
        
        /// <summary>Non-ancestor input operation ambiguity</summary>
        public const string VG012 = "VG012";
        
        /// <summary>Output tensors array cannot be empty</summary>
        public const string VG013 = "VG013";
        
        /// <summary>Input tensors contain invalid variables</summary>
        public const string VG014 = "VG014";
        
        /// <summary>Output tensors contain invalid variables</summary>
        public const string VG015 = "VG015";
        
        /// <summary>Tensor casting to Variable not implemented</summary>
        public const string VG016 = "VG016";
        
        /// <summary>Tensor not found in graph canonical tensors</summary>
        public const string VG017 = "VG017";
        
        /// <summary>Tensor is not in this graph</summary>
        public const string VG018 = "VG018";
        
        /// <summary>Node is not in this graph</summary>
        public const string VG019 = "VG019";
        
        /// <summary>Node is not a graph node</summary>
        public const string VG020 = "VG020";
        
        /// <summary>Node is not in this VirtualGraph</summary>
        public const string VG021 = "VG021";
        
        /// <summary>Directory path is null for file operations</summary>
        public const string VG022 = "VG022";
        
        /// <summary>Recursive parameter required when constant tensors is null</summary>
        public const string VG023 = "VG023";
        
        /// <summary>Cannot inline node as it is not a FunctionNode</summary>
        public const string VG024 = "VG024";
        
        /// <summary>Loop max iteration input must be int64 scalar</summary>
        public const string VG025 = "VG025";
        
        /// <summary>Loop max iteration input must be scalar</summary>
        public const string VG026 = "VG026";
        
        /// <summary>Loop max iteration tensor data invalid</summary>
        public const string VG027 = "VG027";
        
        /// <summary>Loop variable can only be extracted from a loop</summary>
        public const string VG028 = "VG028";
        
        /// <summary>Input length mismatch in rebuild operation</summary>
        public const string VG029 = "VG029";
        
        /// <summary>Output tensors contain null values</summary>
        public const string VG030 = "VG030";
        
        /// <summary>Loop iteration data type must be int64</summary>
        public const string VG031 = "VG031";
        
        /// <summary>Loop iteration data must be scalar</summary>
        public const string VG032 = "VG032";
        
        /// <summary>Input count mismatch for ordered inputs</summary>
        public const string VG033 = "VG033";
        
        /// <summary>UseIfOutput operation not implemented</summary>
        public const string VG034 = "VG034";
        
        #endregion

        #region Global.Constructors Extended Error Codes (GC006-GC020)
        
        /// <summary>Method not implemented for data type</summary>
        public const string GC006 = "GC006";
        
        /// <summary>Invalid operation in global constructor</summary>
        public const string GC007 = "GC007";
        
        /// <summary>Module function null when required</summary>
        public const string GC008 = "GC008";
        
        /// <summary>InputOptionalTensor not implemented</summary>
        public const string GC009 = "GC009";
        
        /// <summary>InputTensorSequence not implemented</summary>
        public const string GC010 = "GC010";
        
        #endregion

        #region Core Error Codes (CR001-CR013)
        
        /// <summary>Invalid IModuleParam type</summary>
        public const string CR001 = "CR001";
        
        /// <summary>TensorDataSequence data cannot be empty when dtype is null</summary>
        public const string CR002 = "CR002";
        
        /// <summary>Shorokoo operation not implemented</summary>
        public const string CR003 = "CR003";
        
        /// <summary>OnnxOps operation not implemented</summary>
        public const string CR004 = "CR004";
        
        /// <summary>Scalar operation not implemented</summary>
        public const string CR005 = "CR005";
        
        /// <summary>ComputeContext operation not implemented</summary>
        public const string CR006 = "CR006";
        
        /// <summary>ComputeContext execution failed with inner exception</summary>
        public const string CR007 = "CR007";

        /// <summary>Variable→handle conversion: structural kind (tensor/optional/sequence/struct) mismatch</summary>
        public const string CR011 = "CR011";

        /// <summary>Variable→handle conversion: element dtype mismatch (use Cast&lt;T&gt;() to convert the dtype)</summary>
        public const string CR012 = "CR012";

        /// <summary>Variable→handle conversion: tensor rank mismatch (scalar/vector handle over a wrongly-ranked node)</summary>
        public const string CR013 = "CR013";

        #endregion

        #region Framework Error Codes (FW001-FW050)
        
        /// <summary>Loop operation not implemented</summary>
        public const string FW001 = "FW001";
        
        /// <summary>ModuleHelper operation not implemented or invalid</summary>
        public const string FW002 = "FW002";
        
        /// <summary>ModuleBase operation not implemented or invalid</summary>
        public const string FW003 = "FW003";
        
        /// <summary>CSharpModelBuilder operation not implemented or invalid</summary>
        public const string FW004 = "FW004";
        
        /// <summary>Function operation not implemented or invalid</summary>
        public const string FW005 = "FW005";
        
        /// <summary>OnnxModelReader operation not implemented or invalid</summary>
        public const string FW006 = "FW006";
        
        /// <summary>NamedModelParam operation not implemented or invalid</summary>
        public const string FW007 = "FW007";
        
        /// <summary>OnnxIRFactory operation not implemented or invalid</summary>
        public const string FW008 = "FW008";
        
        /// <summary>NodeDefinitionMaker operation not implemented or invalid</summary>
        public const string FW009 = "FW009";
        
        /// <summary>Invalid Onnx name provided</summary>
        public const string FW010 = "FW010";
        
        /// <summary>Loop body index validation failed</summary>
        public const string FW011 = "FW011";
        
        /// <summary>Loop body construction failed</summary>
        public const string FW012 = "FW012";
        
        /// <summary>Loop pass OpCode mismatch</summary>
        public const string FW013 = "FW013";
        
        /// <summary>Loop pass input count mismatch</summary>
        public const string FW014 = "FW014";
        
        /// <summary>Loop pass input nullness mismatch</summary>
        public const string FW015 = "FW015";
        
        /// <summary>Loop execution condition invalid</summary>
        public const string FW016 = "FW016";
        
        /// <summary>Loop state variable access invalid</summary>
        public const string FW017 = "FW017";
        
        /// <summary>Loop node sequence validation failed</summary>
        public const string FW018 = "FW018";
        
        /// <summary>Loop step execution failed</summary>
        public const string FW019 = "FW019";
        
        /// <summary>Loop variable binding failed</summary>
        public const string FW020 = "FW020";
        
        /// <summary>Loop output generation failed</summary>
        public const string FW021 = "FW021";
        
        /// <summary>Loop iteration processing failed</summary>
        public const string FW022 = "FW022";

        /// <summary>If-else node input count mismatch</summary>
        public const string FW024 = "FW024";
        
        /// <summary>Custom node with variadic outputs and other outputs not supported</summary>
        public const string FW025 = "FW025";
        
        /// <summary>Custom node with no outputs not supported</summary>
        public const string FW026 = "FW026";
        
        /// <summary>DType inference failed for node definition</summary>
        public const string FW027 = "FW027";
        
        /// <summary>Code template operation not implemented</summary>
        public const string FW028 = "FW028";
        
        /// <summary>Circular reference detected in functions</summary>
        public const string FW029 = "FW029";
        
        /// <summary>Attribute type conversion not implemented</summary>
        public const string FW030 = "FW030";
        
        /// <summary>Sequence element type conversion not implemented</summary>
        public const string FW031 = "FW031";
        
        /// <summary>Data type not supported for operation</summary>
        public const string FW032 = "FW032";
        
        /// <summary>TensorProto data type not supported</summary>
        public const string FW033 = "FW033";
        
        /// <summary>Type conversion method not implemented</summary>
        public const string FW034 = "FW034";
        
        /// <summary>Array element type conversion not implemented</summary>
        public const string FW035 = "FW035";
        
        /// <summary>Optional type conversion not implemented</summary>
        public const string FW036 = "FW036";
        
        /// <summary>Output tensors validation failed</summary>
        public const string FW037 = "FW037";
        
        /// <summary>Output tensors scope validation failed</summary>
        public const string FW038 = "FW038";
        
        /// <summary>Node definition creation operation failed</summary>
        public const string FW039 = "FW039";
        
        /// <summary>Graph scope operation failed</summary>
        public const string FW040 = "FW040";
        
        /// <summary>Virtual graph concrete architecture operation failed</summary>
        public const string FW041 = "FW041";
        
        /// <summary>Parameter initialization failed - concrete architecture required</summary>
        public const string FW042 = "FW042";
        
        /// <summary>IfElse tensor count mismatch</summary>
        public const string FW043 = "FW043";
        
        /// <summary>Utils operation not implemented</summary>
        public const string FW044 = "FW044";
        
        #endregion

        #region Utility Error Codes (UT001-UT010)

        /// <summary>BiDictionary duplicate key or value</summary>
        public const string UT001 = "UT001";

        #endregion

        #region State Update Error Codes (SU001-SU010)

        /// <summary>StateUpdate target is not a state variable created by a [StateInitializer] Init call</summary>
        public const string SU001 = "SU001";

        /// <summary>StateUpdate target is a trainable parameter, not a state variable</summary>
        public const string SU002 = "SU002";

        /// <summary>StateUpdate argument is null or not a graph variable</summary>
        public const string SU003 = "SU003";

        #endregion

        #region AutoDiff Error Codes (AD001-AD025)

        /// <summary>No gradient implementation registered for the op</summary>
        public const string AD001 = "AD001";

        /// <summary>Gradient accumulation is not supported for this variable kind</summary>
        public const string AD002 = "AD002";

        /// <summary>An op on the differentiation path has no registered gradient (or no gradient
        /// for the attribute combination in use) — training through it would silently freeze the
        /// parameters behind it, so the autograd lowering fails loudly instead</summary>
        public const string AD003 = "AD003";

        #endregion
    }
}