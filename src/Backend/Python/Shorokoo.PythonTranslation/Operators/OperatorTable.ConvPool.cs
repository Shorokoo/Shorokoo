namespace Shorokoo.PythonTranslation.Operators;

// Convolution and pooling; the semantics are in each support package's ops_conv_pool.py.
internal static partial class OperatorTable
{
    static partial void RegisterConvPool(Registry table)
    {
        const string M = "ops_conv_pool.";
        string[] convolution = ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"];
        string[] window = ["auto_pad", "ceil_mode", "dilations", "kernel_shape", "pads", "strides"];

        table.Map("Conv", M + "conv", convolution);
        table.Map("ConvTranspose", M + "conv_transpose", [.. convolution, "output_padding", "output_shape"]);
        table.Map("ConvInteger", M + "conv_integer", convolution, gradient: GradientRule.NotDifferentiable);
        table.Map("DeformConv", M + "deform_conv", ["dilations", "group", "kernel_shape", "offset_group", "pads", "strides"]);
        table.Map("MaxPool", M + "max_pool", [.. window, "storage_order"], outputs: true);
        table.Map("AveragePool", M + "average_pool", [.. window, "count_include_pad"]);
        table.Map("LpPool", M + "lp_pool", [.. window, "p"]);
        table.Map("GlobalAveragePool", M + "global_average_pool");
        table.Map("GlobalMaxPool", M + "global_max_pool");
        table.Map("GlobalLpPool", M + "global_lp_pool", ["p"]);
        table.Map("MaxUnpool", M + "max_unpool", ["kernel_shape", "pads", "strides"]);
        table.Map("MaxRoiPool", M + "max_roi_pool", ["pooled_shape", "spatial_scale"]);
    }
}
