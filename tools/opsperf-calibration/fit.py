#!/usr/bin/env python3
"""Fit and score the OpsPerf compute model against ORT profiles.

Input: the per-family JSON files written by ``OpsPerfCalibrationTests`` (run
``dotnet test --filter "FullyQualifiedName~OpsPerfCalibrationTests"`` with
``SHOROKOO_OPSPERF_CALIBRATION_DIR`` pointing at an empty directory).

    python3 tools/opsperf-calibration/fit.py <dir>            # score the C# estimates in the dump, refit, print both
    python3 tools/opsperf-calibration/fit.py <dir> --no-fit   # score only

The model mirrors src/Shorokoo/Core/AutoDiffCheckpointing/OpsPerf/*.cs (ComputeTime in
nanoseconds); ``PARAMS`` holds the constants the C# estimators carry. Refitting prints
the refitted constants; port any you adopt into the C# files by hand and re-run the
harness so the "C# estimate" columns are scored on what actually ships.

Pairing: kernels are paired with nodes by name (the harness records the name each node's
kernel carries). At ORT_ENABLE_ALL a ``FusedMatMul`` kernel replaces a MatMul plus the
Transpose(s) feeding it: each unpaired FusedMatMul is attributed to the kernel-less MatMul
with the same output shape, and a kernel-less Transpose whose only consumer is a MatMul is
scored as fused (measured 0). Kernels inside a Loop body are profiled per iteration and
summed per run; nodes are scored on the per-execution time, and the per-family totals
exclude body kernels (the Loop kernel's own time already contains them).
"""
import json, math, os, sys, glob, collections
import numpy as np

P = lambda d: int(np.prod(d)) if d else 1

# ----------------------------------------------------------------------------- model
# Constants the C# estimators carry (ns). Refit prints replacements.
PARAMS = dict(
    launch=4200.0,             # per-kernel fixed cost (ns)
    copy=0.0175,                # ns per byte moved, contiguous streaming (elementwise, copies)
    copy_mult=3.3,             # multiplier on copy for strided copies (Concat/Slice/Pad/Expand/Gather…)
    broadcast=1.33,             # multiplier when a non-scalar input is broadcast
    transpose_inner=0.070,      # ns/byte when the innermost axis moves
    transpose_outer=0.035,      # ns/byte when it does not
    reduce_inner=0.030,        # ns/byte of input, reduced axes innermost
    reduce_outer=0.113,         # ns/byte of input, a reduced axis is not innermost
    softmax_elem=1.38,          # ns per element
    softmax_row=82.0,         # ns per row (softmax axis vector)
    matmul_flop=0.00363,        # ns per FLOP, asymptotic
    matmul_sqrt=9.67,           # ns per sqrt(FLOP): blocking/threading ramp
    conv_flop=0.0133,           # ns per FLOP, large output image
    conv_spatial=47.7,         # ns/FLOP grows as (1 + conv_spatial / output image size)
    rnn_flop=0.0086,            # ns per FLOP for LSTM/GRU/RNN kernels
    shape_survival=0.0,        # fraction of int64 shape arithmetic ORT does not constant-fold
    meta_survival=0.28,         # fraction of Reshape/Squeeze/Unsqueeze kernels that survive
)
UNARY_MULT = dict(Abs=1, Neg=1, Sign=1, Ceil=1, Floor=1, Round=1, Not=1, BitwiseNot=1, Relu=1.3, LeakyRelu=1.3,
                  Selu=2, Elu=2, Celu=2, Gelu=2.75, Sigmoid=2, Tanh=2, Sqrt=1.5, Reciprocal=1.5, Exp=1.2, Log=1.5,
                  Sin=2, Cos=2, Tan=2, Asin=2, Acos=2, Atan=2, Sinh=2, Cosh=2, Atanh=2, Asinh=2, Acosh=2, Erf=2.35,
                  Cast=1, CastLike=1, IsInf=1, IsNaN=1, Dropout=2, Clip=1)
BINARY_MULT = dict(Add=1, Sub=1, Mul=1, Div=1, Mod=2, Pow=3, Max=1, Min=1, Mean=1, Sum=1, And=1, Or=1, Xor=1,
                   BitwiseAnd=1, BitwiseOr=1, BitwiseXor=1, BitShift=1, Equal=1, Greater=1.45, GreaterOrEqual=1.45,
                   Less=1.45, LessOrEqual=1.45, Where=4.3)
REDUCE = {'ReduceSum', 'ReduceMean', 'ReduceMax', 'ReduceMin', 'ReduceProd', 'ReduceL1', 'ReduceL2', 'ReduceLogSum',
          'ReduceLogSumExp', 'ReduceSumSquare', 'ArgMax', 'ArgMin', 'CumSum'}
COPY = {'Expand', 'Concat', 'Split', 'Slice', 'Pad', 'Tile', 'Gather', 'GatherElements', 'GatherND', 'ScatterElements',
        'ScatterND', 'ConstantOfShape', 'Range', 'OneHot', 'Compress', 'Trilu', 'Resize', 'Upsample', 'DepthToSpace',
        'SpaceToDepth', 'EyeLike', 'NonZero', 'ReverseSequence', 'Unique', 'CenterCropPad'}
META = {'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze'}
ZERO = {'Identity', 'Constant', '#ModelTensorInput#', 'Shape', 'Size', 'SequenceEmpty'}


def bytes_of(t): return t['bytes'] if t else 0
def elems(t): return P(t['dims']) if t else 0


def is_shape_arith(n):
    outs = [o for o in n['outs'] if o]
    return bool(outs) and all(o['dtype'] != 'Float32' and o['bytes'] <= 64 for o in outs)


def model(n, p=PARAMS):
    """ns for one execution of node n — mirrors the C# estimators."""
    op = n['op']; ins = [i for i in n['ins'] if i]; outs = [o for o in n['outs'] if o]
    if op in ZERO or op.startswith('Loop#') or op.startswith('If#'): return 0.0
    if not outs: return 0.0
    L = p['launch']
    if op in META: return p['meta_survival'] * L
    out = outs[0]; ob = bytes_of(out); oe = elems(out)
    if op in ('MatMul', 'FusedMatMul', 'Gemm'):
        a, b = ins[0]['dims'], ins[1]['dims']
        if op == 'Gemm':
            m, k = (a[0], a[1]) if len(a) == 2 else (1, a[0]); nn = b[1] if len(b) == 2 else b[0]
            if n['attrs'].get('transA', '0') != '0': m, k = k, m
            if n['attrs'].get('transB', '0') != '0': nn = b[0]
            f = 2.0 * m * nn * k
        else:
            m = a[-2] if len(a) >= 2 else 1; k = a[-1]; nn = b[-1]
            batch = max(P(a[:-2]) if len(a) > 2 else 1, P(b[:-2]) if len(b) > 2 else 1)
            f = 2.0 * batch * m * nn * k
        return L + p['matmul_flop'] * f + p['matmul_sqrt'] * math.sqrt(f)
    if op in ('Conv', 'ConvTranspose'):
        w = ins[1]['dims']; od = out['dims']; g = int(n['attrs'].get('group', '1'))
        spatial = P(od[2:]); kvol = P(w[2:]); cin = w[1] if op == 'Conv' else w[0] // g
        f = 2.0 * od[0] * od[1] * spatial * kvol * cin
        return L + f * p['conv_flop'] * (1 + p['conv_spatial'] / spatial)
    if op in ('LSTM', 'GRU', 'RNN'):
        d = ins[0]['dims']; seq, batch, isz = d[0], d[1], d[2]; h = int(n['attrs'].get('hidden_size', isz))
        gates = dict(LSTM=4, GRU=3, RNN=1)[op]
        f = seq * gates * (isz * h + h * h) * 2.0 * batch
        return L + p['rnn_flop'] * f
    if op == 'Softmax' or op == 'LogSoftmax' or op == 'Hardmax':
        d = ins[0]['dims']; axis = int(n['attrs'].get('axis', '-1')); axis = axis % len(d) if d else 0
        rowlen = d[axis] if d else 1; rows = max(1, elems(ins[0]) // max(1, rowlen))
        return L + p['softmax_elem'] * elems(ins[0]) + p['softmax_row'] * rows
    if op == 'Transpose':
        if n.get('consumers') and all(c == 'MatMul' for c in n['consumers']):
            perm = n['attrs'].get('perm'); d = ins[0]['dims']
            if perm and len(d) >= 2:
                pm = [int(x) for x in perm.strip('[]').split(',')]
                if pm[-1] == len(d) - 2 and pm[-2] == len(d) - 1 and pm[:-2] == list(range(len(d) - 2)):
                    return 0.0   # fused into FusedMatMul(transA/transB)
        perm = n['attrs'].get('perm'); d = ins[0]['dims']; rank = len(d)
        inner_moves = True
        if perm:
            pm = [int(x) for x in perm.strip('[]').split(',')]; inner_moves = pm[-1] != rank - 1
        rate = p['transpose_inner'] if inner_moves else p['transpose_outer']
        return L + rate * (bytes_of(ins[0]) + ob)
    if op in REDUCE:
        ib = bytes_of(ins[0]); d = ins[0]['dims']; rank = len(d)
        axes = n['attrs'].get('axes')
        if axes is None and len(ins) > 1 and ins[1].get('dims') is not None:
            axes = None  # runtime axes tensor: unknown, infer from shapes below
        ie, oe_ = elems(ins[0]), oe
        if ie == oe_:   # empty axes / no-op reduce: a copy
            return L + p['copy'] * (ib + ob)
        # reduced axes innermost? compare trailing dims of input/output (keepdims either way)
        odims = out['dims']; inner_reduced = True
        if odims and len(odims) == rank:
            inner_reduced = odims[-1] == 1 and d[-1] != 1 or all(x == y for x, y in zip(d, odims))
            # a leading axis reduced while the last is kept -> outer
            inner_reduced = odims[-1] == 1 or d[-1] == 1
        elif odims:
            inner_reduced = len(odims) == 0 or odims[-1] != d[-1]
        rate = p['reduce_inner'] if inner_reduced else p['reduce_outer']
        return L + rate * ib + p['copy'] * ob
    if op in BINARY_MULT:
        moved = sum(bytes_of(i) for i in ins) + ob
        bc = any(0 < elems(i) < oe for i in ins)
        c = L + p['copy'] * BINARY_MULT[op] * moved * (p['broadcast'] if bc else 1)
        return p['shape_survival'] * c if is_shape_arith(n) else c
    if op in UNARY_MULT:
        c = L + p['copy'] * UNARY_MULT[op] * (bytes_of(ins[0]) + ob)
        return p['shape_survival'] * c if is_shape_arith(n) else c
    if op in COPY:
        moved = sum(bytes_of(i) for i in ins) + ob
        c = L + p['copy'] * p['copy_mult'] * moved
        return p['shape_survival'] * c if is_shape_arith(n) else c
    if op in ('AveragePool', 'MaxPool', 'LpPool', 'GlobalAveragePool', 'GlobalMaxPool', 'GlobalLpPool'):
        return L + p['copy'] * (bytes_of(ins[0]) + ob)
    if op in ('BatchNormalization', 'InstanceNormalization', 'GroupNormalization', 'LayerNormalization', 'LpNormalization', 'LRN'):
        return L + p['copy'] * 5 * (bytes_of(ins[0]) + ob)
    return L + p['copy'] * (sum(bytes_of(i) for i in ins) + ob)


# ----------------------------------------------------------------------------- pairing
def load(dir_):
    fams = {}
    for f in sorted(glob.glob(os.path.join(dir_, '*.json'))):
        d = json.load(open(f)); fams[d['family']] = d
    return fams


def pair(G, level):
    """-> list of (node, measured_ns_per_exec, executions) and the family's measured top-level total (ns)."""
    by_name = {n['name']: n for n in G['nodes'] if n['name']}
    kernels = G['kernels'][level]
    have = {k['name'] for k in kernels}
    pairs, total = [], 0.0
    scoped = set()
    for k in kernels:
        n = by_name.get(k['name'])
        if n is not None and n.get('scope'): scoped.add(k['name'])
    for k in kernels:
        if k['name'] not in scoped: total += k['us'] * 1000.0
        n = by_name.get(k['name'])
        if n is not None: pairs.append((n, k['us'] * 1000.0 / k['count'], k['count']))
    unpaired = [k for k in kernels if k['name'] not in by_name]
    # FusedMatMul -> kernel-less MatMul with the same output dims (in order)
    free_mm = [n for n in G['nodes'] if n['op'] == 'MatMul' and n['name'] not in have]
    fused_pairs = 0
    for k in list(unpaired):
        if k['op'] != 'FusedMatMul': continue
        od = None
        try: od = list(next(iter(k['outs'][0].values())))
        except Exception: pass
        for n in free_mm:
            if n['outs'][0] and n['outs'][0]['dims'] == od:
                pairs.append((n, k['us'] * 1000.0 / k['count'], k['count'])); free_mm.remove(n); unpaired.remove(k); fused_pairs += 1
                break
    fused_t = [n for n in G['nodes'] if n['op'] == 'Transpose' and n['name'] not in have
               and n.get('consumers') and all(c == 'MatMul' for c in n['consumers'])]
    real_nokernel = [n for n in G['nodes'] if n['name'] and n['name'] not in have and n['op'] not in ZERO
                     and n not in free_mm and n not in fused_t and not n['op'].startswith('Loop#')]
    return pairs, total, unpaired, fused_pairs, fused_t, real_nokernel, free_mm


# ----------------------------------------------------------------------------- stats
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def score(fams, level, estimator, label):
    allest, allmeas = [], []
    print(f"\n== {label} @ {level}")
    print(f"{'family':12s} {'graph':5s} {'n':>5s} {'spearman':>9s} {'med|log|':>9s} {'sum est':>11s} {'sum meas':>11s} {'ratio':>7s}  unpairedK fusedMM fusedT realNoK")
    fam_ratios = {}
    for fam, d in fams.items():
        for g in ('pre', 'post'):
            G = d['graphs'][g]
            pairs, total, unpaired, fmm, ft, rnk, free_mm = pair(G, level)
            est = np.array([estimator(n) for n, _, _ in pairs]); meas = np.array([m for _, m, _ in pairs])
            ok = (est > 0) & (meas > 0)
            rho = spearman(est[ok], meas[ok]); ml = float(np.median(np.abs(np.log(est[ok] / meas[ok]))))
            sum_est = sum(estimator(n) for n in G['nodes'] if not n.get('scope'))
            # the Loop kernel's own time contains its body; the estimator counts body nodes once
            sum_est += sum(estimator(n) for n in G['nodes'] if n.get('scope'))
            ratio = sum_est / total if total else float('nan')
            if g == 'post': fam_ratios[fam] = ratio
            print(f"{fam:12s} {g:5s} {len(pairs):5d} {rho:9.3f} {ml:9.3f} {sum_est/1e6:9.2f}ms {total/1e6:9.2f}ms {ratio:7.2f}  {len(unpaired):9d} {fmm:7d} {len(ft):6d} {len(rnk):7d}")
            allest.append(est[ok]); allmeas.append(meas[ok])
    e = np.concatenate(allest); m = np.concatenate(allmeas)
    big = m >= 10000
    print(f"ALL nodes: n={len(e)} spearman={spearman(e, m):.3f} median|log(est/meas)|={np.median(np.abs(np.log(e/m))):.3f} "
          f"p90|log|={np.percentile(np.abs(np.log(e/m)), 90):.3f}")
    print(f"nodes measured >= 10 us (above the launch floor): n={int(big.sum())} spearman={spearman(e[big], m[big]):.3f} "
          f"median|log|={np.median(np.abs(np.log(e[big]/m[big]))):.3f} p90|log|={np.percentile(np.abs(np.log(e[big]/m[big])), 90):.3f}")
    worst = max(fam_ratios.values()) if fam_ratios else 0; best = min(fam_ratios.values()) if fam_ratios else 0
    print(f"per-family total ratio (post graphs): min {best:.2f}, max {worst:.2f}; families outside [0.5, 2]: "
          f"{[f for f, r in fam_ratios.items() if not 0.5 <= r <= 2]}")
    return e, m


def per_op(fams, level, estimator):
    byop = collections.defaultdict(lambda: [[], []])
    for fam, d in fams.items():
        for g in ('pre', 'post'):
            pairs, *_ = pair(d['graphs'][g], level)
            for n, m, _ in pairs:
                e = estimator(n)
                if e > 0 and m > 0: byop[n['op']][0].append(e); byop[n['op']][1].append(m)
    print(f"\n{'op':18s} {'n':>5s} {'median log(est/meas)':>21s} {'med|log|':>9s} {'sum est/meas':>13s}")
    for op, (e, m) in sorted(byop.items(), key=lambda kv: -sum(kv[1][1])):
        e = np.array(e); m = np.array(m); lg = np.log(e / m)
        print(f"{op:18s} {len(e):5d} {np.median(lg):21.2f} {np.median(np.abs(lg)):9.2f} {e.sum()/m.sum():13.2f}")


# ----------------------------------------------------------------------------- fitting
def nelder_mead(f, x0, steps, iters=400):
    n = len(x0); pts = [np.array(x0, float)]
    for i in range(n):
        x = np.array(x0, float); x[i] += steps[i]; pts.append(x)
    vals = [f(x) for x in pts]
    for _ in range(iters):
        order = np.argsort(vals); pts = [pts[i] for i in order]; vals = [vals[i] for i in order]
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]: pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (x - pts[0]) for x in pts[1:]]; vals = [vals[0]] + [f(x) for x in pts[1:]]
    i = int(np.argmin(vals)); return pts[i]


def fit_keys(rows, p, keys, pred=None):
    """Minimise sum log^2(model/measured) over p[keys] on rows [(node, ns)]; keys stay positive."""
    if not rows: return
    base = np.array([p[k] for k in keys], float)
    m = np.array([r[1] for r in rows], float)
    def loss(x):
        q = dict(p); q.update({k: float(base[i] * math.exp(x[i])) for i, k in enumerate(keys)})
        e = np.array([model(r[0], q) for r in rows], float)
        return float(np.sum(np.log(np.maximum(e, 1.0) / m) ** 2))
    x = nelder_mead(loss, np.zeros(len(keys)), [0.5] * len(keys))
    p.update({k: float(base[i] * math.exp(x[i])) for i, k in enumerate(keys)})


def fit(fams, level):
    rows = []
    for fam, d in fams.items():
        for g in ('pre', 'post'):
            pairs, *_ = pair(d['graphs'][g], level)
            rows += [(n, m) for n, m, _ in pairs if m > 0]
    p = dict(PARAMS)
    traffic = lambda n: sum(bytes_of(t) for t in n['ins'] + n['outs'] if t)
    sel = lambda pred: [(n, m) for n, m in rows if pred(n)]

    tiny = [m for n, m in rows if traffic(n) <= 256 and n['op'] not in META and n['op'] not in ZERO]
    p['launch'] = float(np.median(tiny))
    for cls, keyset in (('shape', 'shape_survival'), ('meta', 'meta_survival')):
        tot = hit = 0
        for fam, d in fams.items():
            G = d['graphs']['post']; have = {k['name'] for k in G['kernels'][level]}
            for n in G['nodes']:
                is_cls = n['op'] in META if cls == 'meta' else (is_shape_arith(n) and n['op'] not in ZERO and n['op'] not in META)
                if is_cls: tot += 1; hit += n['name'] in have
        if tot: p[keyset] = hit / tot

    plain = lambda n: n['op'] in ('Add', 'Sub', 'Mul') and not is_shape_arith(n) and not any(0 < elems(i) < elems(n['outs'][0]) for i in n['ins'] if i)
    fit_keys(sel(lambda n: plain(n) and traffic(n) >= 16384), p, ['copy'])
    fit_keys(sel(lambda n: n['op'] in ('Add', 'Sub', 'Mul') and not is_shape_arith(n)
                 and any(1 < elems(i) < elems(n['outs'][0]) for i in n['ins'] if i) and traffic(n) >= 16384), p, ['broadcast'])
    for op in ('Div', 'Where', 'Greater', 'Relu', 'Exp', 'Erf', 'Gelu', 'Neg'):
        rs = sel(lambda n, op=op: n['op'] == op and traffic(n) >= 16384)
        if len(rs) < 3: continue
        table = BINARY_MULT if op in BINARY_MULT else UNARY_MULT
        m = np.array([x for _, x in rs]); base = table[op]
        def loss(x):
            table[op] = base * math.exp(x[0]); return float(np.sum(np.log(np.array([max(model(n, p), 1.0) for n, _ in rs]) / m) ** 2))
        x = nelder_mead(loss, [0.0], [0.5]); table[op] = round(base * math.exp(x[0]), 2)
    inner = lambda n: (not n['attrs'].get('perm')) or [int(x) for x in n['attrs']['perm'].strip('[]').split(',')][-1] != len(n['ins'][0]['dims']) - 1
    fit_keys(sel(lambda n: n['op'] == 'Transpose' and inner(n) and traffic(n) >= 16384), p, ['transpose_inner'])
    fit_keys(sel(lambda n: n['op'] == 'Transpose' and not inner(n) and traffic(n) >= 16384), p, ['transpose_outer'])
    def red_inner(n):
        d = n['ins'][0]['dims']; od = n['outs'][0]['dims']
        return (od[-1] == 1 or d[-1] == 1) if len(od) == len(d) else (len(od) == 0 or od[-1] != d[-1])
    red = lambda n: n['op'] in ('ReduceSum', 'ReduceMean') and elems(n['ins'][0]) != elems(n['outs'][0]) and traffic(n) >= 16384
    fit_keys(sel(lambda n: red(n) and red_inner(n)), p, ['reduce_inner'])
    fit_keys(sel(lambda n: red(n) and not red_inner(n)), p, ['reduce_outer'])
    fit_keys(sel(lambda n: n['op'] in COPY and not is_shape_arith(n) and traffic(n) >= 16384), p, ['copy_mult'])
    fit_keys(sel(lambda n: n['op'] == 'Softmax'), p, ['softmax_elem', 'softmax_row'])
    fit_keys(sel(lambda n: n['op'] == 'MatMul'), p, ['matmul_flop', 'matmul_sqrt'])
    fit_keys(sel(lambda n: n['op'] in ('Conv', 'ConvTranspose')), p, ['conv_flop', 'conv_spatial'])
    fit_keys(sel(lambda n: n['op'] in ('LSTM', 'GRU', 'RNN')), p, ['rnn_flop'])
    return p


def main():
    dir_ = sys.argv[1]; refit = '--no-fit' not in sys.argv
    fams = load(dir_)
    cs = lambda n: n['est']
    print("#### C# estimates in the dump (whatever units the estimators currently use)")
    score(fams, 'enableAll', cs, 'C# estimate as dumped')
    per_op(fams, 'enableAll', cs)
    if not refit: return
    p = fit(fams, 'enableAll')
    print("\n#### refitted parameters")
    for k, v in p.items(): print(f"  {k:18s} = {v:.5g}")
    print("  unary multipliers:", {k: v for k, v in UNARY_MULT.items() if v != 1})
    print("  binary multipliers:", {k: v for k, v in BINARY_MULT.items() if v != 1})
    est = lambda n: model(n, p)
    score(fams, 'enableAll', est, 'python model, refitted')
    per_op(fams, 'enableAll', est)
    score(fams, 'disableAll', est, 'python model, refitted')
    print("\nMatMul nodes measured > 3x the model (per family, post graph):")
    for fam, d in fams.items():
        pairs, *_ = pair(d['graphs']['post'], 'enableAll')
        for n, m, _ in pairs:
            if n['op'] == 'MatMul' and m > 3 * est(n):
                print(f"  {fam:12s} {n['ins'][0]['dims']}x{n['ins'][1]['dims']} measured={m/1000:.0f}us model={est(n)/1000:.0f}us")


if __name__ == '__main__':
    main()
