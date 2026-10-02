"""The memory-reuse scenario on PyTorch (eagerly, on the CPU and on a card) and on JAX/XLA (on the
CPU), each way it can be written, and through Shorokoo's translation of the scenario's ONNX model
for each. Driven by MemoryReuseScenarioTests, which hands it a JSON file naming the sizes, the
translations, the support packages and where to write the figures.

A run's figures are what it allocated beyond its two inputs: on a card, torch's caching allocator's
own counters; on the CPU, the allocations torch's profiler records; for XLA, the buffer assignment
of the compiled program (memory_analysis), and the inputs donation consumed."""

import base64
import ctypes
import json
import sys
import time
import traceback

import numpy as np

config = json.load(open(sys.argv[1], encoding="utf-8"))
N, M = config["N"], config["M"]
MIB = 1 << 20
results = {"N": N, "M": M, "torch": [], "jax": []}


def mib(value):
    return round(value / MIB, 2)


def expected_values():
    i = np.arange(2 * N * M, dtype=np.int64)
    a = ((i % 1013).astype(np.float32) * np.float32(0.001)).reshape(2 * N, M)
    b = (np.float32(0.5) - (i % 997).astype(np.float32) * np.float32(0.002)).reshape(2 * N, M)
    sig = lambda x: 1 / (1 + np.exp(-np.asarray(x, dtype=np.float64)))
    c = sig(sig(2.0))
    l = np.concatenate([np.full((N, M), c), sig(b[N:])])
    return a, b, l


A0, B0, L0 = expected_values()


def correct(l, a_half, b_half):
    return bool(np.array_equal(a_half, A0[:N]) and np.array_equal(b_half, B0[N:])
                and np.allclose(l, L0, atol=1e-5))


# ---- torch ---------------------------------------------------------------------------------------

def run_torch():
    import torch
    sys.path.insert(0, config["torch_package"])
    from shorokoo_torch import runtime as trt

    def naive(A, B):
        """Out of place, as an eager program reads: the halves are views, every op a new tensor."""
        n = A.shape[0] // 2
        A_half, B_half = A[:n], B[n:]
        C = torch.full(A_half.shape, 2.0, device=A.device)
        C = torch.sigmoid(torch.abs(torch.neg(C)))
        L = torch.cat([C, B_half], 0)
        del C
        L = torch.abs(torch.neg(torch.sigmoid(L)))
        return L, A_half, B_half

    def in_place(A, B):
        """The unary ops in place; the fill and the concatenation each a new tensor."""
        n = A.shape[0] // 2
        A_half, B_half = A[:n], B[n:]
        C = torch.full(A_half.shape, 2.0, device=A.device)
        C.neg_().abs_().sigmoid_()
        L = torch.cat([C, B_half], 0)
        del C
        L.sigmoid_().neg_().abs_()
        return L, A_half, B_half

    def cat_out_overlapping(A, B):
        """cat(out=B) with B_half still in B: torch refuses an output overlapping an input."""
        n = A.shape[0] // 2
        C = torch.full((n, A.shape[1]), 2.0, device=A.device)
        C.neg_().abs_().sigmoid_()
        torch.cat([C, B[n:]], 0, out=B)
        B.sigmoid_().neg_().abs_()
        return B, A[:n], A[:n]

    def cat_out(A, B):
        """B_half compacted into A first, then cat(out=B): the fill is the one new tensor."""
        n = A.shape[0] // 2
        A[n:].copy_(B[n:])
        A_half, B_half = A[:n], A[n:]
        C = torch.full((n, A.shape[1]), 2.0, device=A.device)
        C.neg_().abs_().sigmoid_()
        torch.cat([C, B_half], 0, out=B)
        del C
        B.sigmoid_().neg_().abs_()
        return B, A_half, B_half

    def ideal(A, B):
        """Views and in-place ops only: B_half compacted into A's second half, C made in B's first
        half, so that B holds the concatenation as it stands, and every op in place."""
        n = A.shape[0] // 2
        A[n:].copy_(B[n:])
        A_half, B_half = A[:n], A[n:]
        C = B[:n]
        C.fill_(2.0).neg_().abs_().sigmoid_()
        L = B
        L.sigmoid_().neg_().abs_()
        return L, A_half, B_half

    def load_translation(device):
        constants = []
        for constant in config["torch_constants"]:
            data = base64.b64decode(constant["bytes"])
            value = torch.empty(tuple(constant["shape"]), dtype=trt.torch_dtype(constant["code"]))
            if data:
                ctypes.memmove(value.data_ptr(), data, len(data))
            constants.append(value.to(device))
        main = trt.load_model(config["torch_source"], "<memory-reuse-scenario>", constants)
        storages = trt.constant_storages(constants)
        ids = trt.constant_ids(constants)

        def translated(A, B):
            outputs = trt.run(main, [A, B], [0, 1, 2], str(device), storages, ids)
            return tuple(value for value, _, _ in outputs)
        return translated

    def inputs(device):
        A = torch.from_numpy(A0.copy()).to(device)
        B = torch.from_numpy(B0.copy()).to(device)
        return A, B

    def placement(outputs, a_ptr, b_ptr):
        whole = 2 * N * M * 4
        names = []
        for value in outputs:
            p = value.data_ptr()
            if a_ptr <= p < a_ptr + whole:
                names.append(f"A+{(p - a_ptr) // MIB}MiB")
            elif b_ptr <= p < b_ptr + whole:
                names.append(f"B+{(p - b_ptr) // MIB}MiB")
            else:
                names.append("own")
        return " / ".join(names)

    def measure_cuda(name, fn, device):
        record = {"backend": "torch", "device": "cuda", "variant": name}
        try:
            for _ in range(3):
                A, B = inputs(device)
                a_ptr, b_ptr = A.data_ptr(), B.data_ptr()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                before = torch.cuda.memory_stats(device)
                start = time.perf_counter()
                outputs = fn(A, B)
                del A, B
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                after = torch.cuda.memory_stats(device)
                record.update({
                    "allocations": int(after["allocation.all.allocated"] - before["allocation.all.allocated"]),
                    "bytes_allocated_mib": mib(after["allocated_bytes.all.allocated"] - before["allocated_bytes.all.allocated"]),
                    "peak_beyond_inputs_mib": mib(torch.cuda.max_memory_allocated(device) - base),
                    "held_after_inputs_released_mib": mib(torch.cuda.memory_allocated(device) - base),
                    "outputs": placement(outputs, a_ptr, b_ptr),
                    "correct": correct(*[o.cpu().numpy() for o in outputs]),
                    "ms": round(elapsed * 1000, 2),
                })
                del outputs
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        results["torch"].append(record)

    def measure_cpu(name, fn):
        from torch.profiler import profile, ProfilerActivity
        record = {"backend": "torch", "device": "cpu", "variant": name}
        try:
            for _ in range(3):
                A, B = inputs("cpu")
                a_ptr, b_ptr = A.data_ptr(), B.data_ptr()
                start = time.perf_counter()
                with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
                    outputs = fn(A, B)
                    del A, B
                elapsed = time.perf_counter() - start
                in_use = peak = total = count = 0
                # An operator's own allocations are its self figure (its children's are theirs); a
                # release outside every operator is a "[memory]" event of its own.
                for event in sorted(prof.events(), key=lambda e: e.time_range.start):
                    change = event.cpu_memory_usage if event.name == "[memory]" else event.self_cpu_memory_usage
                    if change == 0:
                        continue
                    in_use += change
                    peak = max(peak, in_use)
                    if change > 0:
                        total += change
                        if change >= MIB:
                            count += 1
                record.update({
                    "allocations": count,
                    "bytes_allocated_mib": mib(total),
                    "peak_beyond_inputs_mib": mib(peak),
                    "held_after_inputs_released_mib": mib(in_use),
                    "outputs": placement(outputs, a_ptr, b_ptr),
                    "correct": correct(*[o.numpy() for o in outputs]),
                    "ms": round(elapsed * 1000, 2),
                })
                del outputs
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        results["torch"].append(record)

    variants = [("naive", naive), ("in-place", in_place), ("cat-out-overlapping", cat_out_overlapping),
                ("cat-out", cat_out), ("ideal", ideal)]
    with torch.no_grad():
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            for name, fn in variants + [("shorokoo-translation", load_translation(device))]:
                measure_cuda(name, fn, device)
        for name, fn in variants + [("shorokoo-translation", load_translation(torch.device("cpu")))]:
            measure_cpu(name, fn)


# ---- JAX -----------------------------------------------------------------------------------------

def run_jax():
    import jax
    import jax.numpy as jnp
    from jax.sharding import SingleDeviceSharding
    sys.path.insert(0, config["jax_package"])
    from shorokoo_jax import runtime as jrt

    cpu = jax.devices("cpu")[0]

    def scenario(A, B):
        n = A.shape[0] // 2
        A_half, B_half = A[:n], B[n:]
        C = jnp.full(A_half.shape, 2.0, A.dtype)
        C = jax.nn.sigmoid(jnp.abs(-C))
        L = jnp.concatenate([C, B_half], 0)
        L = jnp.abs(-jax.nn.sigmoid(L))
        return L, A_half, B_half

    def scenario_runtime_fill(A, B):
        """The fill's value read from the data, so that XLA cannot fold C's chain to a constant."""
        n = A.shape[0] // 2
        A_half, B_half = A[:n], B[n:]
        two = A[0, 0] * 0 + 2
        C = jnp.broadcast_to(two, A_half.shape)
        C = jax.nn.sigmoid(jnp.abs(-C))
        L = jnp.concatenate([C, B_half], 0)
        L = jnp.abs(-jax.nn.sigmoid(L))
        return L, A_half, B_half

    def scenario_halves_together(A, B):
        """The halves handed back as one [2N, M] output, the shape of a donated input, so that XLA
        may write them into A's buffer as it may write L into B's."""
        L, A_half, B_half = scenario(A, B)
        return L, jnp.concatenate([A_half, B_half], 0)

    def scenario_halves_into_a(A, B):
        """The same with the halves first, so that donation pairs them with A and L with B: the
        ideal's placement, B_half written into A's second half and L over B."""
        L, A_half, B_half = scenario(A, B)
        return jnp.concatenate([A_half, B_half], 0), L

    def aliases(compiled):
        text = compiled.as_text()
        start = text.find("input_output_alias={")
        if start < 0:
            return ""
        depth, end = 0, start + len("input_output_alias=")
        while end < len(text):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            end += 1
            if depth == 0:
                break
        return text[start:end]

    # Inputs XLA allocated itself, as a run's inputs are: an array put on the CPU from numpy may be
    # numpy's own memory, which donation cannot hand to an output.
    fresh = jax.jit(jnp.copy)
    hlo_dir = config["hlo_dir"]

    def analyse(name, compiled, call, donated):
        with open(f"{hlo_dir}/xla-{name}-{'donated' if donated else 'kept'}.hlo.txt", "w", encoding="utf-8") as hlo:
            hlo.write(compiled.as_text())
        stats = compiled.memory_analysis()
        record = {
            "backend": "jax", "device": "cpu", "variant": name, "donated": donated,
            "argument_mib": mib(stats.argument_size_in_bytes),
            "output_mib": mib(stats.output_size_in_bytes),
            "alias_mib": mib(stats.alias_size_in_bytes),
            "temp_mib": mib(stats.temp_size_in_bytes),
            "beyond_inputs_mib": mib(stats.output_size_in_bytes - stats.alias_size_in_bytes + stats.temp_size_in_bytes),
            "aliases": aliases(compiled),
        }
        try:
            for _ in range(3):
                A = jax.block_until_ready(fresh(jax.device_put(A0, cpu)))
                B = jax.block_until_ready(fresh(jax.device_put(B0, cpu)))
                a_ptr = A.unsafe_buffer_pointer()
                b_ptr = B.unsafe_buffer_pointer()
                start = time.perf_counter()
                outputs = jax.block_until_ready(call(A, B))
                elapsed = time.perf_counter() - start
                record["inputs_deleted"] = f"{A.is_deleted()} / {B.is_deleted()}"
                whole = 2 * N * M * 4
                places = []
                for value in outputs:
                    p = value.unsafe_buffer_pointer()
                    places.append(f"A+{(p - a_ptr) // MIB}MiB" if a_ptr <= p < a_ptr + whole
                                  else f"B+{(p - b_ptr) // MIB}MiB" if b_ptr <= p < b_ptr + whole else "own")
                record["outputs"] = " / ".join(places)
                arrays = [np.asarray(o) for o in outputs]
                if len(arrays) == 2:
                    halves, l = (arrays[0], arrays[1]) if name == "halves-into-a" else (arrays[1], arrays[0])
                    arrays = [l, halves[:N], halves[N:]]
                record["correct"] = correct(*arrays)
                record["ms"] = round(elapsed * 1000, 2)
                del outputs, A, B
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        results["jax"].append(record)

    spec = jax.ShapeDtypeStruct((2 * N, M), np.float32, sharding=SingleDeviceSharding(cpu))
    for name, fn in [("scenario", scenario), ("runtime-fill", scenario_runtime_fill),
                     ("halves-together", scenario_halves_together), ("halves-into-a", scenario_halves_into_a)]:
        for donated in (False, True):
            jitted = jax.jit(fn, donate_argnums=(0, 1) if donated else ())
            compiled = jitted.lower(spec, spec).compile()
            analyse(name, compiled, compiled, donated)

    # Shorokoo's translation, compiled as its JAX backend compiles it, and the same with the two
    # inputs donated.
    constants = []
    for constant in config["jax_constants"]:
        data = base64.b64decode(constant["bytes"])
        constants.append(np.frombuffer(data, dtype=jrt.jax_dtype(constant["code"])).reshape(tuple(constant["shape"])).copy())
    model = jrt.load_model(config["jax_source"], "<memory-reuse-scenario>", constants, "cpu")
    signature = (((2 * N, M), np.dtype(np.float32)), ((2 * N, M), np.dtype(np.float32)))
    program = model.program(signature)
    analyse("shorokoo-translation", program, lambda A, B: program(jrt._fresh_key(), model._large_values, A, B), False)
    sharding = SingleDeviceSharding(cpu)
    key = jax.ShapeDtypeStruct((2,), np.uint32, sharding=sharding)
    large = [jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=sharding) for v in model._large_values]
    donated = jax.jit(model._entry, donate_argnums=(2, 3)).lower(key, large, spec, spec).compile()
    analyse("shorokoo-translation", donated, lambda A, B: donated(jrt._fresh_key(), model._large_values, A, B), True)


for part in (run_torch, run_jax):
    try:
        part()
    except Exception:
        results.setdefault("errors", []).append(traceback.format_exc())

json.dump(results, open(config["out"], "w", encoding="utf-8"), indent=1)
