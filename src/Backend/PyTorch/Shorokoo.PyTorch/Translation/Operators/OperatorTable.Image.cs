namespace Shorokoo.PyTorch.Translation.Operators;

// Image and geometry, and the block rearrangements; the semantics are in shorokoo_torch/ops_image.py.
internal static partial class OperatorTable
{
    static partial void RegisterImage(Registry table)
    {
        const string M = "ops_image.";
        table.Map("Resize", M + "resize",
        [
            "antialias", "axes", "coordinate_transformation_mode", "cubic_coeff_a", "exclude_outside",
            "extrapolation_value", "keep_aspect_ratio_policy", "mode", "nearest_mode",
        ], opset: true);
        table.Map("Upsample", M + "upsample", ["mode"]);
        table.Map("GridSample", M + "grid_sample", ["align_corners", "mode", "padding_mode"]);
        table.Map("AffineGrid", M + "affine_grid", ["align_corners"]);
        table.Map("RoiAlign", M + "roi_align",
        [
            "coordinate_transformation_mode", "mode", "output_height", "output_width", "sampling_ratio", "spatial_scale",
        ]);
        table.Map("NonMaxSuppression", M + "non_max_suppression", ["center_point_box"],
            gradient: TorchGradient.NotDifferentiable);
        table.Map("CenterCropPad", M + "center_crop_pad", ["axes"]);
        table.Map("Col2Im", M + "col2im", ["dilations", "pads", "strides"]);
        table.Map("DepthToSpace", M + "depth_to_space", ["blocksize", "mode"]);
        table.Map("SpaceToDepth", M + "space_to_depth", ["blocksize"]);
        table.Map("ImageDecoder", M + "image_decoder", ["pixel_format"], gradient: TorchGradient.NotDifferentiable);
    }
}
