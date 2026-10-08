// The operators Shorokoo adds to ONNX Runtime's CPU execution provider, registered on a session's
// options through RegisterCustomOps (OrtApi::RegisterCustomOpsLibrary_V2 calls it by that name).
//
// AdamUpdate, in the ai.shorokoo domain, is one Adam or AdamW step over one float32 parameter, in a
// single pass: it reads the parameter, its gradient and both moments once and writes the parameter
// and the moments once. Written out as the ONNX operators the optimizer is built of, the same step
// is twelve passes over the parameter (thirteen with a weight decay), each streaming a tensor of
// its size through memory, so it costs in proportion to the parameter count whatever the model's
// arithmetic. OrtFusedUpdates.cs substitutes it for exactly that chain of operators, and this
// computes what the chain computes, operation for operation in float32 and in the chain's order,
// so the result is the same to the bit:
//
//   m' = beta1 * m + c1 * g
//   v' = beta2 * v + (c2 * g) * g
//   p' = p [* decay] - m' / (sqrt(v') + eps) * step
//
// Inputs: p, m, v, g of one shape; beta1, c1, beta2, c2, eps, step, and optionally decay, of one
// element each. Outputs: p', m', v', of p's shape. Output k may be bound to the memory of input k
// (k < 3): each element is read before the same element is written.
//
// Like the allocator, it links nothing of ONNX Runtime's: the runtime that registers it hands it
// the API it calls, and each runtime in the process gets an operator and a domain of its own, which
// call that runtime's API and no other.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <vector>

#include "onnxruntime_c_api.h"

#if defined(_WIN32)
#define SHOROKOO_EXPORT extern "C" __declspec(dllexport)
#else
#define SHOROKOO_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

// The API version asked of the runtime: the one that first has every entry point used here
// (KernelContext_ParallelFor, the shape-inference context, CreateKernelV2 and KernelComputeV2).
constexpr uint32_t ApiVersion = 17;

constexpr const char* Domain = "ai.shorokoo";

// Elements per task handed to the runtime's thread pool: large enough that a task's overhead is
// nothing beside streaming it, small enough to spread a parameter of a few hundred thousand
// elements over every core.
constexpr size_t Chunk = 1 << 14;

enum Input : size_t { P, M, V, G, Beta1, C1, Beta2, C2, Eps, Step, Decay, InputCount };
constexpr size_t OutputCount = 3;

struct Update {
    const float* p;
    const float* m;
    const float* v;
    const float* g;
    float* pOut;
    float* mOut;
    float* vOut;
    float beta1, c1, beta2, c2, eps, step, decay;
    bool decays;
    size_t count;
};

// Element by element, each value read before the same element is written, so an output may be
// the memory of the input it replaces. No contraction into fused multiply-adds (see CMakeLists):
// each operation rounds as the operator it stands for does. An output is the memory of its input
// or apart from every input, so no element depends on another and the loop is vectorized as such;
// told so, since the compiler cannot see it and would otherwise run an output written in place one
// element at a time.
#if defined(_MSC_VER)
#define SHOROKOO_INDEPENDENT __pragma(loop(ivdep))
#else
#define SHOROKOO_INDEPENDENT _Pragma("GCC ivdep")
#endif

template <bool Decays>
void Run(void* state, size_t task) {
    const auto& u = *static_cast<const Update*>(state);
    const size_t begin = task * Chunk;
    const size_t end = begin + Chunk < u.count ? begin + Chunk : u.count;
    SHOROKOO_INDEPENDENT
    for (size_t i = begin; i < end; ++i) {
        const float g = u.g[i];
        const float m = u.beta1 * u.m[i] + u.c1 * g;
        const float v = u.beta2 * u.v[i] + (u.c2 * g) * g;
        const float p = Decays ? u.p[i] * u.decay : u.p[i];
        const float step = m / (std::sqrt(v) + u.eps) * u.step;
        u.mOut[i] = m;
        u.vOut[i] = v;
        u.pOut[i] = p - step;
    }
}

// The operator as one runtime holds it, with that runtime's API: what the entry points below are
// handed back is the OrtCustomOp, the first member.
struct Op {
    OrtCustomOp ort;
    const OrtApi* api;
    OrtCustomOpDomain* domain;
};

const OrtApi* ApiOf(const OrtCustomOp* op) { return reinterpret_cast<const Op*>(op)->api; }

// A kernel holds the API of the runtime that made it.
struct Kernel {
    const OrtApi* api;
};

#define SHOROKOO_TRY(call)                       \
    do {                                         \
        if (OrtStatus* failed = (call)) return failed; \
    } while (0)

// The shape and element count of a tensor value.
struct Shape {
    int64_t dims[16];
    size_t rank = 0;
    size_t count = 0;
    bool isFloat = false;
};

OrtStatus* Fail(const OrtApi* api, const char* message) { return api->CreateStatus(ORT_INVALID_ARGUMENT, message); }

OrtStatus* ShapeOf(const OrtApi* api, const OrtValue* value, Shape& shape) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    SHOROKOO_TRY(api->GetTensorTypeAndShape(value, &info));
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    OrtStatus* status = api->GetTensorElementType(info, &type);
    if (status == nullptr) status = api->GetDimensionsCount(info, &shape.rank);
    if (status == nullptr && shape.rank > 16) status = Fail(api, "AdamUpdate takes tensors of rank 16 at most.");
    if (status == nullptr) status = api->GetDimensions(info, shape.dims, shape.rank);
    if (status == nullptr) status = api->GetTensorShapeElementCount(info, &shape.count);
    api->ReleaseTensorTypeAndShapeInfo(info);
    shape.isFloat = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    return status;
}

bool SameShape(const Shape& a, const Shape& b) {
    if (a.rank != b.rank) return false;
    for (size_t d = 0; d < a.rank; ++d)
        if (a.dims[d] != b.dims[d]) return false;
    return true;
}

OrtStatus* Tensor(const OrtApi* api, OrtKernelContext* context, size_t index, const Shape& like, const float*& data) {
    const OrtValue* value = nullptr;
    SHOROKOO_TRY(api->KernelContext_GetInput(context, index, &value));
    if (value == nullptr) return Fail(api, "AdamUpdate's parameter, moments and gradient are required.");
    Shape shape;
    SHOROKOO_TRY(ShapeOf(api, value, shape));
    if (!shape.isFloat || !SameShape(shape, like))
        return Fail(api, "AdamUpdate's parameter, moments and gradient must be float32 tensors of one shape.");
    const void* raw = nullptr;
    SHOROKOO_TRY(api->GetTensorData(value, &raw));
    data = static_cast<const float*>(raw);
    return nullptr;
}

OrtStatus* Single(const OrtApi* api, OrtKernelContext* context, size_t index, float& out, bool& present) {
    const OrtValue* value = nullptr;
    SHOROKOO_TRY(api->KernelContext_GetInput(context, index, &value));
    present = value != nullptr;
    if (!present) return nullptr;
    Shape shape;
    SHOROKOO_TRY(ShapeOf(api, value, shape));
    if (!shape.isFloat || shape.count != 1) return Fail(api, "AdamUpdate's coefficients must be float32 values of one element.");
    const void* raw = nullptr;
    SHOROKOO_TRY(api->GetTensorData(value, &raw));
    out = *static_cast<const float*>(raw);
    return nullptr;
}

OrtStatus* Output(const OrtApi* api, OrtKernelContext* context, size_t index, const Shape& shape, float*& data) {
    OrtValue* value = nullptr;
    SHOROKOO_TRY(api->KernelContext_GetOutput(context, index, shape.dims, shape.rank, &value));
    void* raw = nullptr;
    SHOROKOO_TRY(api->GetTensorMutableData(value, &raw));
    data = static_cast<float*>(raw);
    return nullptr;
}

OrtStatus* ORT_API_CALL Compute(void* kernel, OrtKernelContext* context) {
    const OrtApi* api = static_cast<Kernel*>(kernel)->api;
    const OrtValue* parameter = nullptr;
    SHOROKOO_TRY(api->KernelContext_GetInput(context, P, &parameter));
    if (parameter == nullptr) return Fail(api, "AdamUpdate's parameter is required.");
    Shape shape;
    SHOROKOO_TRY(ShapeOf(api, parameter, shape));
    if (!shape.isFloat) return Fail(api, "AdamUpdate updates float32 parameters.");

    Update u{};
    u.count = shape.count;
    SHOROKOO_TRY(Tensor(api, context, P, shape, u.p));
    SHOROKOO_TRY(Tensor(api, context, M, shape, u.m));
    SHOROKOO_TRY(Tensor(api, context, V, shape, u.v));
    SHOROKOO_TRY(Tensor(api, context, G, shape, u.g));
    bool present = false;
    float* coefficients[] = {&u.beta1, &u.c1, &u.beta2, &u.c2, &u.eps, &u.step};
    for (size_t k = 0; k < 6; ++k) {
        SHOROKOO_TRY(Single(api, context, Beta1 + k, *coefficients[k], present));
        if (!present) return Fail(api, "AdamUpdate's coefficients other than the decay are required.");
    }
    SHOROKOO_TRY(Single(api, context, Decay, u.decay, u.decays));
    SHOROKOO_TRY(Output(api, context, 0, shape, u.pOut));
    SHOROKOO_TRY(Output(api, context, 1, shape, u.mOut));
    SHOROKOO_TRY(Output(api, context, 2, shape, u.vOut));
    if (u.count == 0) return nullptr;

    const size_t tasks = (u.count + Chunk - 1) / Chunk;
    auto* run = u.decays ? &Run<true> : &Run<false>;
    if (tasks == 1) {
        run(&u, 0);
        return nullptr;
    }
    return api->KernelContext_ParallelFor(context, run, tasks, 0, &u);
}

OrtStatus* ORT_API_CALL CreateKernel(const OrtCustomOp* op, const OrtApi*, const OrtKernelInfo*, void** kernel) {
    auto* made = new (std::nothrow) Kernel{ApiOf(op)};
    if (made == nullptr) return ApiOf(op)->CreateStatus(ORT_FAIL, "There was no memory for an AdamUpdate kernel.");
    *kernel = made;
    return nullptr;
}

void ORT_API_CALL DestroyKernel(void* kernel) { delete static_cast<Kernel*>(kernel); }

const char* ORT_API_CALL Name(const OrtCustomOp*) { return "AdamUpdate"; }

const char* ORT_API_CALL Provider(const OrtCustomOp*) { return "CPUExecutionProvider"; }

ONNXTensorElementDataType ORT_API_CALL FloatType(const OrtCustomOp*, size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL InputCountOf(const OrtCustomOp*) { return InputCount; }

size_t ORT_API_CALL OutputCountOf(const OrtCustomOp*) { return OutputCount; }

OrtCustomOpInputOutputCharacteristic ORT_API_CALL InputCharacteristic(const OrtCustomOp*, size_t index) {
    return index == Decay ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL OutputCharacteristic(const OrtCustomOp*, size_t) {
    return INPUT_OUTPUT_REQUIRED;
}

OrtMemType ORT_API_CALL InputMemory(const OrtCustomOp*, size_t) { return OrtMemTypeDefault; }

int ORT_API_CALL NotVariadic(const OrtCustomOp*) { return 1; }

// Output k is input k's type and shape, so the graph the runtime writes out states every value. The
// context owns the input's type and shape it hands out, and releases it itself.
OrtStatus* ORT_API_CALL InferShape(const OrtCustomOp* op, OrtShapeInferContext* context) {
    const OrtApi* api = ApiOf(op);
    for (size_t k = 0; k < OutputCount; ++k) {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        SHOROKOO_TRY(api->ShapeInferContext_GetInputTypeShape(context, k, &info));
        SHOROKOO_TRY(api->ShapeInferContext_SetOutputTypeShape(context, k, info));
    }
    return nullptr;
}

int ORT_API_CALL StartVersion(const OrtCustomOp*) { return 1; }

int ORT_API_CALL EndVersion(const OrtCustomOp*) { return 1; }

OrtCustomOp MakeAdamUpdate() {
    OrtCustomOp op{};
    op.version = ApiVersion;
    op.GetName = &Name;
    op.GetExecutionProviderType = &Provider;
    op.GetInputType = &FloatType;
    op.GetInputTypeCount = &InputCountOf;
    op.GetOutputType = &FloatType;
    op.GetOutputTypeCount = &OutputCountOf;
    op.KernelDestroy = &DestroyKernel;
    op.GetInputCharacteristic = &InputCharacteristic;
    op.GetOutputCharacteristic = &OutputCharacteristic;
    op.GetInputMemoryType = &InputMemory;
    op.GetVariadicInputMinArity = &NotVariadic;
    op.GetVariadicInputHomogeneity = &NotVariadic;
    op.GetVariadicOutputMinArity = &NotVariadic;
    op.GetVariadicOutputHomogeneity = &NotVariadic;
    op.CreateKernelV2 = &CreateKernel;
    op.KernelComputeV2 = &Compute;
    op.InferOutputShapeFn = &InferShape;
    op.GetStartVersion = &StartVersion;
    op.GetEndVersion = &EndVersion;
    return op;
}

// One operator and domain per runtime, for the life of the process: a session's options hold the
// domain by reference, and a session built from them reads it for as long as it lives.
std::mutex Gate;
std::vector<Op*> Ops;

}  // namespace

// Adds the ai.shorokoo domain to `options`, for the runtime `base` belongs to. ONNX Runtime calls
// this when the library is registered on a session's options. A runtime older than the API version
// asked for gets nothing added, and a model holding the operator then fails to build its session;
// the managed side registers it only with the runtime it ships.
SHOROKOO_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
    const OrtApi* api = base->GetApi(ApiVersion);
    if (api == nullptr) return nullptr;
    Op* registered = nullptr;
    {
        std::lock_guard<std::mutex> lock(Gate);
        for (Op* op : Ops)
            if (op->api == api) registered = op;
        if (registered == nullptr) {
            auto* op = new (std::nothrow) Op{MakeAdamUpdate(), api, nullptr};
            if (op == nullptr) return api->CreateStatus(ORT_FAIL, "There was no memory for Shorokoo's operators.");
            if (OrtStatus* failed = api->CreateCustomOpDomain(Domain, &op->domain)) {
                delete op;
                return failed;
            }
            if (OrtStatus* failed = api->CustomOpDomain_Add(op->domain, &op->ort)) {
                api->ReleaseCustomOpDomain(op->domain);
                delete op;
                return failed;
            }
            Ops.push_back(op);
            registered = op;
        }
    }
    return api->AddCustomOpDomain(options, registered->domain);
}
