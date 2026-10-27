// The OrtAllocator ONNX Runtime calls for every session Shorokoo builds, over Shorokoo's own
// allocator, which is managed.
//
// The managed allocator answers through [UnmanagedCallersOnly] entry points, and those can return
// a null block but cannot refuse one any other way: nothing may unwind out of them. A null is no
// refusal to ONNX Runtime. It does not check an allocator's answer, hands the null to the kernel
// that asked, and the kernel's first write is an illegal access. Its own allocators refuse by
// throwing a C++ exception, which it catches at the kernel or at its C API boundary and turns into
// a failed call. So this is the allocator ONNX Runtime holds: it forwards each request to the
// managed side, and where that answers null it throws std::runtime_error carrying the reason the
// managed side wrote down, as ONNX Runtime's own allocators do.
//
// It needs nothing of ONNX Runtime's beyond the layout of OrtAllocator, which the C API fixes: a
// version, then entry points, a later version only ever appending. So it links nothing of ONNX
// Runtime and serves whichever runtime a process loads, any number of them side by side; it holds
// no state of its own beyond each allocator it makes.

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>

#if defined(_WIN32)
#define SHOROKOO_EXPORT extern "C" __declspec(dllexport)
#else
#define SHOROKOO_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct OrtMemoryInfo;

namespace {

// OrtAllocator as onnxruntime_c_api.h lays it out. ORT_API_CALL is the platform's one calling
// convention on every 64-bit target, so the entry points are plain function pointers.
struct OrtAllocator {
    uint32_t version;
    void* (*Alloc)(OrtAllocator* self, size_t size);
    void (*Free)(OrtAllocator* self, void* block);
    const OrtMemoryInfo* (*Info)(const OrtAllocator* self);
    void* (*Reserve)(OrtAllocator* self, size_t size);
    void* (*GetStats)(const OrtAllocator* self, void** out);
    void* (*AllocOnStream)(OrtAllocator* self, size_t size, void* stream);
    void* (*Shrink)(OrtAllocator* self);
};

// The version announced: the fields up to Shrink, of which Alloc, Free, Info and AllocOnStream are
// filled in. ONNX Runtime calls an optional entry point only where it is not null.
constexpr uint32_t AnnouncedVersion = 25;

// The managed allocator's entry points. Allocate answers a block, or null with the reason written
// into `reason` (NUL-terminated, at most `capacity` bytes); AllocateOnStream does the same for a
// request ONNX Runtime makes on one of its streams, which it names. None ever unwinds.
using ManagedAllocate = void* (*)(void* state, size_t size, char* reason, int32_t capacity);
using ManagedAllocateOnStream = void* (*)(void* state, size_t size, void* stream, char* reason, int32_t capacity);
using ManagedFree = void (*)(void* state, void* block);

// What ONNX Runtime holds: the OrtAllocator first, so the pointer it is handed back is this.
struct Allocator {
    OrtAllocator ort;
    void* state;
    ManagedAllocate allocate;
    ManagedAllocateOnStream allocateOnStream;
    ManagedFree free;
    const OrtMemoryInfo* info;
};

// Long enough for any reason the managed side words, and on the stack, so that refusing takes no
// memory beyond the exception's own.
constexpr int32_t ReasonCapacity = 1024;

// The block, or the refusal the managed side worded, thrown as ONNX Runtime's own allocators throw.
void* Answer(void* block, size_t size, char* reason) {
    if (block != nullptr || size == 0) return block;
    reason[ReasonCapacity - 1] = '\0';
    throw std::runtime_error(reason[0] != '\0' ? reason : "Failed to allocate: Shorokoo's allocator refused the request.");
}

void* Alloc(OrtAllocator* self, size_t size) {
    auto* allocator = reinterpret_cast<Allocator*>(self);
    char reason[ReasonCapacity];
    reason[0] = '\0';
    return Answer(allocator->allocate(allocator->state, size, reason, ReasonCapacity), size, reason);
}

// A request on one of ONNX Runtime's streams. Implementing it is what makes ONNX Runtime name the
// stream: otherwise it asks through Alloc, and the managed side cannot tell a request whose work the
// stream orders from one ONNX Runtime fills off every stream.
void* AllocOnStream(OrtAllocator* self, size_t size, void* stream) {
    auto* allocator = reinterpret_cast<Allocator*>(self);
    char reason[ReasonCapacity];
    reason[0] = '\0';
    return Answer(allocator->allocateOnStream(allocator->state, size, stream, reason, ReasonCapacity), size, reason);
}

void Free(OrtAllocator* self, void* block) {
    auto* allocator = reinterpret_cast<Allocator*>(self);
    allocator->free(allocator->state, block);
}

const OrtMemoryInfo* Info(const OrtAllocator* self) {
    return reinterpret_cast<const Allocator*>(self)->info;
}

}  // namespace

// An OrtAllocator over the managed one `state` names, describing itself to ONNX Runtime with `info`
// (an OrtMemoryInfo), which the caller keeps alive for as long as the allocator is used. Null where there is no memory
// for it. It lives for the life of the process: an allocator registered with ONNX Runtime's
// environment cannot be taken back while anything it made is alive.
SHOROKOO_EXPORT void* shorokoo_ort_allocator_create(
    void* state, ManagedAllocate allocate, ManagedAllocateOnStream allocateOnStream, ManagedFree free,
    const void* info) {
    auto* allocator = new (std::nothrow) Allocator{};
    if (allocator == nullptr) return nullptr;
    allocator->ort.version = AnnouncedVersion;
    allocator->ort.Alloc = &Alloc;
    allocator->ort.Free = &Free;
    allocator->ort.AllocOnStream = &AllocOnStream;
    allocator->ort.Info = &Info;
    allocator->state = state;
    allocator->allocate = allocate;
    allocator->allocateOnStream = allocateOnStream;
    allocator->free = free;
    allocator->info = static_cast<const OrtMemoryInfo*>(info);
    return &allocator->ort;
}
