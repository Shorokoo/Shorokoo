using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Tests;

// Reference-generation harness for ModuleForwardValueTests (Purpose=Manual so it never runs
// in the coverage suite — run it explicitly to (re)generate frozen constants). Emits:
//   * golden values (collapsed Shorokoo forward output) for self-generated references, and
//   * a .safetensors export of the seeded weights + input for the PyTorch references
//     (consumed by tests/pytorch-reference/*.py).
// Set PILOT_DIR to control the output directory (default /tmp/pilot). As more modules are
// converted, add their golden/export lines here alongside the two examples below.
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
public class _GoldenReferenceGen
{
    [Fact]
    public void Generate()
    {
        string dir = Environment.GetEnvironmentVariable("PILOT_DIR") ?? "/tmp/pilot";
        Directory.CreateDirectory(dir);

        // --- golden: Conv2d collapsed forward output ---
        var conv = ModuleForwardValueTests.Collapse(
            ModuleForwardValueTests.RunForward<ParityConv2d>(ModuleForwardValueTests.SinInput([2L, 2L, 9L, 9L])));
        Console.WriteLine("GOLDEN_CONV2D " + string.Join(", ", conv.Select(v => v.ToString("R") + "f")));

        // --- pytorch: export Linear seeded weights + input for tests/pytorch-reference/linear.py ---
        var input = ModuleForwardValueTests.SinInput([3L, 5L]);
        var g = ParityLinear.ComputationGraph;
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));
        var pl = arch.InitializeTrainableParams(rngConfig: ModuleForwardValueTests.ParitySeed);
        var tensors = new List<SafeTensor>();
        foreach (var p in pl.ModelParams)
        {
            var td = p.ToTensorData();
            tensors.Add(new SafeTensor(p.ParamName, td, "F32", td.Shape.Dims.ToArray()));
        }
        tensors.Add(new SafeTensor("input", input, "F32", input.Shape.Dims.ToArray()));
        SafeTensorLoader.SaveSafeTensors(Path.Combine(dir, "linear.safetensors"), tensors);
        Console.WriteLine("WROTE " + Path.Combine(dir, "linear.safetensors"));
    }
}
