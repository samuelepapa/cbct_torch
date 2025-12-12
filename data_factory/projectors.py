import torch
import torchbeam as tb
from torch.autograd import Function


def _add_photon_noise(projections, photon_count, torch_rng):
    noisy_data = torch.poisson(torch.exp(-projections) * photon_count, generator=torch_rng)
    noisy_data = torch.clamp(noisy_data, min=1.0) / photon_count
    projections = -torch.log(noisy_data)

    return projections


def add_photon_noise(projections, photon_count, torch_rng):
    if photon_count is not None:
        if isinstance(projections, list):
            for i in range(len(projections)):
                projections[i] = _add_photon_noise(
                    torch.tensor(projections[i]), photon_count, torch_rng
                ).numpy()
        elif isinstance(projections, torch.Tensor):
            projections = _add_photon_noise(projections, photon_count, torch_rng)
    else:
        pass
    return projections


def _add_gaussian_noise(projections, mean, std_dev, torch_rng):
    noise = (
        torch.randn(projections.shape, device=projections.device, generator=torch_rng) * std_dev
        + mean
    )
    return projections + noise


def add_gaussian_noise(projections, mean, std_dev, torch_rng):
    if mean is not None and std_dev is not None:
        if isinstance(projections, list):
            for i in range(len(projections)):
                projections[i] = _add_gaussian_noise(
                    torch.tensor(projections[i]), mean, std_dev, torch_rng
                ).numpy()
        elif isinstance(projections, torch.Tensor):
            projections = _add_gaussian_noise(projections, mean, std_dev, torch_rng)
    return projections


class CBProjector(Function):
    @staticmethod
    def forward(
        ctx,
        volume,
        vol_bbox,
        src_points,
        det_center,
        det_frame,
        det_bbox,
        vol_sz,
        det_sz,
        sampling_step_size,
    ):
        if volume.is_cuda:
            assert "cb_proj_forward_cuda" in dir(tb), (
                "TorchBeam was compiled without CUDA support, "
                "but CUDA Torch tensor given as input"
            )
            output = tb.cb_proj_forward_cuda(
                volume,
                vol_bbox,
                src_points,
                det_center,
                det_frame,
                det_bbox,
                det_sz,
                sampling_step_size,
            )
        else:
            output = tb.cb_proj_forward_cpu(
                volume,
                vol_bbox,
                src_points,
                det_center,
                det_frame,
                det_bbox,
                det_sz,
                sampling_step_size,
            )
        ctx.vol_bbox = vol_bbox
        ctx.src_points = src_points
        ctx.det_center = det_center
        ctx.det_frame = det_frame
        ctx.det_bbox = det_bbox
        ctx.vol_sz = vol_sz
        ctx.det_sz = det_sz
        return output

    @staticmethod
    def backward(ctx, grad_input):
        vol_bbox = ctx.vol_bbox
        src_points = ctx.src_points
        det_center = ctx.det_center
        det_frame = ctx.det_frame
        det_bbox = ctx.det_bbox
        vol_sz = ctx.vol_sz
        if grad_input.is_cuda:
            assert "cb_backproj_forward_cuda" in dir(tb), (
                "TorchBeam was compiled without CUDA support, "
                "but CUDA Torch tensor given as input"
            )
            output = tb.cb_backproj_forward_cuda(
                grad_input, vol_bbox, src_points, det_center, det_frame, det_bbox, vol_sz, 1, 1.0
            )
        else:
            output = tb.cb_backproj_forward_cpu(
                grad_input, vol_bbox, src_points, det_center, det_frame, det_bbox, vol_sz, 1, 1.0
            )
        return output, None, None, None, None, None, None, None, None


class CBBackprojector(Function):
    @staticmethod
    def forward(
        ctx,
        projection,
        vol_bbox,
        src_points,
        det_center,
        det_frame,
        det_bbox,
        vol_sz,
        det_sz,
        sampling_step_size,
    ):
        if projection.is_cuda:
            assert "cb_backproj_forward_cuda" in dir(tb), (
                "TorchBeam was compiled without CUDA support, "
                "but CUDA Torch tensor provided as input"
            )
            output = tb.cb_backproj_forward_cuda(
                projection, vol_bbox, src_points, det_center, det_frame, det_bbox, vol_sz, 1, 1.0
            )
        else:
            output = tb.cb_backproj_forward_cpu(
                projection, vol_bbox, src_points, det_center, det_frame, det_bbox, vol_sz, 1, 1.0
            )
        ctx.vol_bbox = vol_bbox
        ctx.src_points = src_points
        ctx.det_center = det_center
        ctx.det_frame = det_frame
        ctx.det_bbox = det_bbox
        ctx.vol_sz = vol_sz
        ctx.det_sz = det_sz
        ctx.sampling_step_size = sampling_step_size
        return output

    @staticmethod
    def backward(ctx, grad_input):
        vol_bbox = ctx.vol_bbox
        src_points = ctx.src_points
        det_center = ctx.det_center
        det_frame = ctx.det_frame
        det_bbox = ctx.det_bbox
        det_sz = ctx.det_sz
        sampling_step_size = ctx.sampling_step_size
        if grad_input.is_cuda:
            assert "cb_proj_forward_cuda" in dir(tb), (
                "TorchBeam was compiled without CUDA support, "
                "but CUDA Torch tensor given as input"
            )
            output = tb.cb_proj_forward_cuda(
                grad_input,
                vol_bbox,
                src_points,
                det_center,
                det_frame,
                det_bbox,
                det_sz,
                sampling_step_size,
            )
        else:
            output = tb.cb_proj_forward_cpu(
                grad_input,
                vol_bbox,
                src_points,
                det_center,
                det_frame,
                det_bbox,
                det_sz,
                sampling_step_size,
            )
        return output, None, None, None, None, None, None, None, None


class CBBackprojectorNoWeight(Function):
    @staticmethod
    def forward(
        ctx,
        projection,
        vol_bbox,
        src_points,
        det_center,
        det_frame,
        det_bbox,
        vol_sz,
        det_sz,
        sampling_step_size,
    ):
        if projection.is_cuda:
            assert "cb_backproj_forward_cuda" in dir(tb), (
                "TorchBeam was compiled without CUDA support, "
                "but CUDA Torch tensor provided as input"
            )
            output = tb.cb_backproj_forward_cuda(
                projection, vol_bbox, src_points, det_center, det_frame, det_bbox, vol_sz, 0, 1.0
            )
        else:
            output = tb.cb_backproj_forward_cpu(
                projection, vol_bbox, src_points, det_center, det_frame, det_bbox, vol_sz, 0, 1.0
            )
        ctx.vol_bbox = vol_bbox
        ctx.src_points = src_points
        ctx.det_center = det_center
        ctx.det_frame = det_frame
        ctx.det_bbox = det_bbox
        ctx.vol_sz = vol_sz
        ctx.det_sz = det_sz
        ctx.sampling_step_size = sampling_step_size
        return output

    @staticmethod
    def backward(ctx, grad_input):
        # No gradients for this function
        return None, None, None, None, None, None, None, None, None


def norm_estimate(
    backproj,
    vol_bbox,
    src_points,
    det_centers,
    det_frames,
    det_bbox,
    vol_sz,
    detector_sz,
    max_iter=5,
    multiplier=1.0,
    vnorm="l2",
):
    with torch.no_grad():

        def tensor_norm(t, norm):
            if norm == "l2":
                v = torch.sqrt(torch.sum(t * t, (2, 3, 4), keepdim=True))
            elif norm == "inf":
                v = torch.amax(torch.abs(t), (2, 3, 4), keepdim=True)
            else:
                raise NotImplementedError
            return v

        norm = tensor_norm(backproj, vnorm)
        x = backproj / norm
        opnorms = torch.zeros_like(norm)
        for i in range(max_iter):
            x = CBProjector.apply(
                x, vol_bbox, src_points, det_centers, det_frames, det_bbox, vol_sz, detector_sz, 1.0
            )
            x = CBBackprojector.apply(
                x, vol_bbox, src_points, det_centers, det_frames, det_bbox, vol_sz, detector_sz, 1.0
            )
            norm = tensor_norm(x, vnorm)
            if torch.isnan(norm).any():
                break
            opnorms = torch.maximum(opnorms, torch.sqrt(norm))
            x = x / norm
        if torch.isnan(opnorms).any():
            sys.exit()
        return 1.0 / (opnorms * multiplier)
