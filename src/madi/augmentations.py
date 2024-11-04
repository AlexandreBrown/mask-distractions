import os
import kornia
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as TF

places_dataloader = None
places_iter = None


def strong_augment(obs, data_dir, method, overlay_alpha=0.5):
    """Alter the observation with a strong augmentation."""
    if method == "conv":
        obs_aug = random_conv(obs.clone())
    elif method == "overlay":
        obs_aug = random_overlay(
            obs.clone(), data_dir, method="default", alpha=overlay_alpha
        )
    elif method == "splice":
        obs_aug = random_overlay(obs.clone(), data_dir, method="splice")
    elif method == "none":
        obs_aug = obs.clone()
        # should actually skip call to strong_augment func when args.augment is none
        # but since we rarely use 'none', we prefer leaving out such an if statement
    else:
        raise NotImplementedError(
            "--augment must be one of [conv, overlay, splice, none]"
        )
    return obs_aug


def _load_places(data_dir, batch_size=256, image_size=96, num_workers=4, use_val=False):
    global places_dataloader, places_iter
    # print the environ var DMCGB_DATASETS
    # print('DMCGB_DATASETS:', os.environ['DMCGB_DATASETS'])

    partition = "val" if use_val else "train"
    print(f"Loading {partition} partition of places365_standard...")
    assert os.path.exists(data_dir)
    fp = os.path.join(data_dir, "places365_standard", partition)
    if not os.path.exists(fp):
        print(f"Warning: path {fp} does not exist, falling back to {data_dir}")
        fp = data_dir
    places_dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            fp,
            TF.Compose(
                [
                    TF.RandomResizedCrop(image_size),
                    TF.RandomHorizontalFlip(),
                    TF.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    places_iter = iter(places_dataloader)
    if places_iter is None:
        raise FileNotFoundError(
            "failed to find places365 data at any of the specified paths"
        )
    print("Loaded dataset from", data_dir)


def _get_places_batch(batch_size):
    global places_iter
    try:
        imgs, _ = next(places_iter)
        if imgs.size(0) < batch_size:
            places_iter = iter(places_dataloader)
            imgs, _ = next(places_iter)
    except StopIteration:
        places_iter = iter(places_dataloader)
        imgs, _ = next(places_iter)
    return imgs.cuda()


def random_overlay(
    x,
    data_dir,
    method="default",
    dataset="places365_standard",
    alpha=0.5,
    hue=0.0,
    sat=0.0,
    val=0.6,
):
    """Randomly overlay an image from Places"""
    global places_iter

    channels = 3
    batch_size, frames_channels, height, width = x.shape
    num_frames = frames_channels // channels

    if dataset == "places365_standard":
        if places_dataloader is None:
            _load_places(data_dir, batch_size=x.size(0), image_size=x.size(-1))
        imgs = (
            _get_places_batch(batch_size=x.size(0)).repeat(1, num_frames, 1, 1) * 255.0
        )
    else:
        raise NotImplementedError(
            f'overlay-{method} has not been implemented for dataset "{dataset}"'
        )

    if method == "splice":
        # Create a tensor of HSV thresholds with the same shape as the image
        thresholds = torch.FloatTensor([hue, sat, val]).to(x.device) * 255.0
        thresholds = thresholds.view(1, -1, 1, 1).repeat(
            batch_size, channels, height, width
        )

        # Chunk the input to frames, convert each frame to HSV and concatenate
        x_chunked = torch.chunk(x, channels, dim=1)
        x_hsv = [kornia.color.rgb_to_hsv(chunk) for chunk in x_chunked]
        x_hsv = torch.cat(x_hsv, dim=1)

        # Create a mask of HSV values above the given thresholds. This should be able to detect the agent
        mask = x_hsv > thresholds
        mask = mask.view(batch_size, num_frames, channels, height, width)

        # Apply torch.all along the channel dimension to get a mask per frame
        mask = torch.all(mask, dim=2, keepdim=True).repeat(1, 1, channels, 1, 1)
        mask = mask.view(batch_size, channels * num_frames, height, width)

        # Apply the agent pixels on top of the sampled image
        imgs[mask] = x[mask]
    elif method == "default":
        imgs = imgs * alpha + x * (1 - alpha)
    else:
        raise NotImplementedError(
            f'Overlay type augmentation method "{method}" not implemented'
        )
    return imgs.clamp(0, 255)


def random_conv(x):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    n, c, h, w = x.shape
    for i in range(n):
        weights = torch.randn(3, 3, 3, 3).to(x.device)
        temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
        temp_x = F.pad(temp_x, pad=[1] * 4, mode="replicate")
        out = torch.sigmoid(F.conv2d(temp_x, weights)) * 255.0
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    return total_out.reshape(n, c, h, w)


def batch_from_obs(obs, batch_size=32):
    """Copy a single observation along the batch dimension"""
    if isinstance(obs, torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.repeat(batch_size, 1, 1, 1)

    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, axis=0)
    return np.repeat(obs, repeats=batch_size, axis=0)


def prepare_pad_batch(obs, next_obs, action, batch_size=32):
    """Prepare batch for self-supervised policy adaptation at test-time"""
    batch_obs = batch_from_obs(torch.from_numpy(obs).cuda(), batch_size)
    batch_next_obs = batch_from_obs(torch.from_numpy(next_obs).cuda(), batch_size)
    batch_action = torch.from_numpy(action).cuda().unsqueeze(0).repeat(batch_size, 1)

    return random_crop_cuda(batch_obs), random_crop_cuda(batch_next_obs), batch_action


def identity(x):
    return x


def random_shift(imgs, pad=4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    return kornia.augmentation.RandomCrop((h, w))(imgs)


def random_crop(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (
        w1 is not None and h1 is not None
    ), "must either specify both w1 and h1 or neither of them"
    assert isinstance(x, torch.Tensor) and x.is_cuda, "input must be CUDA tensor"

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(
        x.shape
    ), "window_shape must be a tuple with same number of dimensions as x"

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3),
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


def vit_crop(imgs, size=96, pad=6):
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    return kornia.augmentation.RandomCrop((size, size))(imgs)
