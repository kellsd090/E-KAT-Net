from __future__ import annotations
import csv
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required. Install it with "
        "'pip install opencv-python-headless'."
    ) from exc


LABEL_TO_INDEX = {"benign": 0, "malignant": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_SIZE = 96
OFFICIAL_FOLDS = {1, 2, 3, 4, 5}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _normalize_pathology(value: object, row_number: int) -> str:
    pathology = str(value).strip().lower()
    aliases = {
        "benign": "benign",
        "b": "benign",
        "0": "benign",
        "malignant": "malignant",
        "m": "malignant",
        "1": "malignant",
    }
    if pathology not in aliases:
        raise ValueError(
            f"Unsupported Pathology value {value!r} at CSV row {row_number}"
        )
    return aliases[pathology]


def discover_busbra_images(
    root_dir: str | Path,
    csv_filename: str = "5-fold-cv.csv",
) -> list[Dict[str, object]]:
    """Load BUS-BRA image records, labels, and official fold assignments."""
    root = Path(root_dir)
    images_dir = root / "Images"
    csv_path = root / csv_filename

    if not root.is_dir():
        raise FileNotFoundError(f"BUS-BRA root directory does not exist: {root}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"BUS-BRA image directory does not exist: {images_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"BUS-BRA CSV does not exist: {csv_path}")

    image_by_stem: Dict[str, Path] = {}
    for path in images_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        key = path.stem.strip().lower()
        if key in image_by_stem:
            raise ValueError(
                f"Duplicate image stem: {image_by_stem[key]} and {path}"
            )
        image_by_stem[key] = path

    records: list[Dict[str, object]] = []
    seen_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        columns = {name.strip().lower(): name for name in reader.fieldnames}
        required_columns = {"id", "pathology", "kfold"}
        missing_columns = required_columns.difference(columns)
        if missing_columns:
            raise ValueError(
                f"CSV is missing columns {sorted(missing_columns)}; "
                f"found {reader.fieldnames}"
            )

        id_column = columns["id"]
        pathology_column = columns["pathology"]
        fold_column = columns["kfold"]

        for row_number, row in enumerate(reader, start=2):
            image_id = str(row.get(id_column, "")).strip()
            if not image_id:
                raise ValueError(f"Empty image ID at CSV row {row_number}")

            key = Path(image_id).stem.lower()
            if key in seen_ids:
                raise ValueError(f"Duplicate image ID {image_id!r} at row {row_number}")
            seen_ids.add(key)

            pathology = _normalize_pathology(
                row.get(pathology_column, ""), row_number
            )

            try:
                fold = int(float(str(row.get(fold_column, "")).strip()))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid kFold value at CSV row {row_number}: "
                    f"{row.get(fold_column)!r}"
                ) from exc

            if fold not in OFFICIAL_FOLDS:
                raise ValueError(
                    f"kFold must be between 1 and 5; found {fold} "
                    f"at CSV row {row_number}"
                )

            image_path = image_by_stem.get(key)
            if image_path is None:
                raise FileNotFoundError(
                    f"No image in {images_dir} matches CSV ID {image_id!r}"
                )

            records.append(
                {
                    "image_path": image_path,
                    "image_id": image_path.stem,
                    "case_id": f"BUSBRA/{image_path.stem}",
                    "source_dataset": "BUSBRA",
                    "pathology": pathology,
                    "label": LABEL_TO_INDEX[pathology],
                    "fold": fold,
                }
            )

    unused_images = sorted(set(image_by_stem).difference(seen_ids))
    if unused_images:
        raise ValueError(
            f"Found {len(unused_images)} images without CSV rows; "
            f"examples: {unused_images[:5]}"
        )
    if not records:
        raise RuntimeError(f"No BUS-BRA records were loaded from {csv_path}")

    return records


def split_busbra_official_fold(
    records: Sequence[Dict[str, object]],
    test_fold: int,
) -> Tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    """Use one official fold for testing and the other four for training."""
    if test_fold not in OFFICIAL_FOLDS:
        raise ValueError("test_fold must be one of {1, 2, 3, 4, 5}")

    train_records = [
        dict(record) for record in records if int(record["fold"]) != test_fold
    ]
    test_records = [
        dict(record) for record in records if int(record["fold"]) == test_fold
    ]

    if not train_records or not test_records:
        raise RuntimeError(f"Fold {test_fold} produced an empty split")

    train_ids = {str(record["case_id"]) for record in train_records}
    test_ids = {str(record["case_id"]) for record in test_records}
    overlap = train_ids.intersection(test_ids)
    if overlap:
        raise RuntimeError(f"Train/test leakage detected: {sorted(overlap)[:5]}")

    for split_name, split_records in (
        ("training", train_records),
        ("test", test_records),
    ):
        labels = {int(record["label"]) for record in split_records}
        if labels != {0, 1}:
            raise RuntimeError(
                f"The {split_name} split does not contain both classes: {labels}"
            )

    return train_records, test_records


class CLAHEGrayscale:
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=tile_grid_size,
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        return Image.fromarray(self.clahe.apply(gray))


class AspectRatioMeanPadResize:
    def __init__(self, canvas_size: int = IMAGE_SIZE):
        self.canvas_size = int(canvas_size)
        if self.canvas_size <= 0:
            raise ValueError("canvas_size must be positive")

    def __call__(self, image: Image.Image) -> Tuple[Image.Image, int]:
        gray = image.convert("L")
        width, height = gray.size
        if width <= 0 or height <= 0:
            raise ValueError("Encountered an empty image")

        scale = self.canvas_size / max(width, height)
        resized_width = min(
            self.canvas_size, max(1, int(round(width * scale)))
        )
        resized_height = min(
            self.canvas_size, max(1, int(round(height * scale)))
        )

        is_downsampling = resized_width <= width and resized_height <= height
        interpolation = (
            Image.Resampling.BOX if is_downsampling else Image.Resampling.BICUBIC
        )
        resized = gray.resize(
            (resized_width, resized_height), resample=interpolation
        )

        fill_value = int(round(float(np.asarray(gray, dtype=np.float32).mean())))
        fill_value = int(np.clip(fill_value, 0, 255))
        canvas = Image.new(
            "L", (self.canvas_size, self.canvas_size), color=fill_value
        )
        left = (self.canvas_size - resized_width) // 2
        top = (self.canvas_size - resized_height) // 2
        canvas.paste(resized, (left, top))
        return canvas, fill_value


class ScharrEdgeExtractor:
    def __init__(
        self,
        bilateral_diameter: int = 15,
        bilateral_sigma_color: float = 0.23,
        bilateral_sigma_space: float = 10.0,
        gaussian_sigma: float = 0.55,
        response_percentile: float = 45.0,
        robust_percentile: float = 90.0,
    ):
        self.bilateral_diameter = int(bilateral_diameter)
        self.bilateral_sigma_color = float(bilateral_sigma_color)
        self.bilateral_sigma_space = float(bilateral_sigma_space)
        self.gaussian_sigma = float(gaussian_sigma)
        self.response_percentile = float(response_percentile)
        self.robust_percentile = float(robust_percentile)

        if self.bilateral_diameter <= 0 or self.bilateral_diameter % 2 == 0:
            raise ValueError("bilateral_diameter must be a positive odd integer")
        if self.bilateral_sigma_color <= 0.0:
            raise ValueError("bilateral_sigma_color must be positive")
        if self.bilateral_sigma_space <= 0.0:
            raise ValueError("bilateral_sigma_space must be positive")
        if self.gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be positive")
        if not 0.0 < self.response_percentile < self.robust_percentile <= 100.0:
            raise ValueError(
                "Require 0 < response_percentile < robust_percentile <= 100"
            )

    @staticmethod
    def _non_maximum_suppression(
        magnitude: np.ndarray,
        angle_degrees: np.ndarray,
    ) -> np.ndarray:
        angle = np.mod(angle_degrees, 180.0)
        thinned = np.zeros_like(magnitude, dtype=np.float32)

        left = np.roll(magnitude, 1, axis=1)
        right = np.roll(magnitude, -1, axis=1)
        up = np.roll(magnitude, 1, axis=0)
        down = np.roll(magnitude, -1, axis=0)
        up_right = np.roll(up, -1, axis=1)
        down_left = np.roll(down, 1, axis=1)
        up_left = np.roll(up, 1, axis=1)
        down_right = np.roll(down, -1, axis=1)

        directions = (
            ((angle < 22.5) | (angle >= 157.5), left, right),
            ((angle >= 22.5) & (angle < 67.5), up_right, down_left),
            ((angle >= 67.5) & (angle < 112.5), up, down),
            ((angle >= 112.5) & (angle < 157.5), up_left, down_right),
        )

        for direction_mask, neighbor_a, neighbor_b in directions:
            local_maximum = (
                direction_mask
                & (magnitude >= neighbor_a)
                & (magnitude >= neighbor_b)
                & ((magnitude > neighbor_a) | (magnitude > neighbor_b))
            )
            thinned[local_maximum] = magnitude[local_maximum]

        thinned[[0, -1], :] = 0.0
        thinned[:, [0, -1]] = 0.0
        return thinned

    def __call__(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        else:
            array = np.asarray(image, dtype=np.float32)
            if array.max(initial=0.0) > 1.0:
                array = array / 255.0

        denoised = cv2.bilateralFilter(
            array,
            d=self.bilateral_diameter,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
            borderType=cv2.BORDER_REFLECT101,
        )
        smoothed = cv2.GaussianBlur(
            denoised,
            ksize=(0, 0),
            sigmaX=self.gaussian_sigma,
            sigmaY=self.gaussian_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
        gradient_x = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0)
        gradient_y = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1)
        magnitude = cv2.magnitude(gradient_x, gradient_y)
        angle = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)
        thinned = self._non_maximum_suppression(magnitude, angle)

        cutoff = float(np.percentile(magnitude, self.response_percentile))
        scale = float(np.percentile(magnitude, self.robust_percentile))
        if (
            not np.isfinite(cutoff)
            or not np.isfinite(scale)
            or scale <= cutoff + 1.0e-8
        ):
            normalized = np.zeros_like(magnitude, dtype=np.float32)
        else:
            normalized = np.clip(
                (thinned - cutoff) / (scale - cutoff), 0.0, 1.0
            ).astype(np.float32)

        return torch.from_numpy(normalized).unsqueeze(0)


class BUSBRATransform:
    def __init__(
        self,
        train: bool,
        channel_mean: Optional[Sequence[float]] = None,
        channel_std: Optional[Sequence[float]] = None,
    ):
        self.train = bool(train)
        self.channel_mean = (
            tuple(channel_mean) if channel_mean is not None else None
        )
        self.channel_std = tuple(channel_std) if channel_std is not None else None

        if (self.channel_mean is None) != (self.channel_std is None):
            raise ValueError("channel_mean and channel_std must be supplied together")
        if self.channel_mean is not None:
            if len(self.channel_mean) != 2 or len(self.channel_std) != 2:
                raise ValueError(
                    "Statistics must contain grayscale and edge values"
                )

        self.contrast = CLAHEGrayscale(
            clip_limit=2.0, tile_grid_size=(8, 8)
        )
        self.resize_and_pad = AspectRatioMeanPadResize(IMAGE_SIZE)
        self.edge_extractor = ScharrEdgeExtractor(
            bilateral_diameter=15,
            bilateral_sigma_color=0.23,
            bilateral_sigma_space=10.0,
            gaussian_sigma=0.55,
            response_percentile=45.0,
            robust_percentile=90.0,
        )
        self.intensity_jitter = transforms.ColorJitter(
            brightness=0.10, contrast=0.10
        )

    def deterministic_stages(
        self,
        image: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Image.Image, int]:
        grayscale = image.convert("L")
        enhanced = self.contrast(grayscale)
        fixed_canvas, fill_value = self.resize_and_pad(enhanced)
        return grayscale, enhanced, fixed_canvas, fill_value

    def _augment(self, image: Image.Image, fill_value: int) -> Image.Image:
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)

        angle, translations, scale, shear = transforms.RandomAffine.get_params(
            degrees=(-10.0, 10.0),
            translate=(0.05, 0.05),
            scale_ranges=(0.90, 1.10),
            shears=None,
            img_size=list(image.size),
        )
        image = TF.affine(
            image,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=fill_value,
        )
        return self.intensity_jitter(image)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        _, _, fixed_canvas, fill_value = self.deterministic_stages(image)
        processed = (
            self._augment(fixed_canvas, fill_value)
            if self.train
            else fixed_canvas
        )

        grayscale_tensor = TF.to_tensor(processed)
        edge_tensor = self.edge_extractor(processed)
        image_channels = torch.cat(
            [grayscale_tensor, edge_tensor], dim=0
        )

        if self.channel_mean is not None and self.channel_std is not None:
            mean = torch.tensor(
                self.channel_mean, dtype=image_channels.dtype
            ).view(2, 1, 1)
            std = torch.tensor(
                self.channel_std, dtype=image_channels.dtype
            ).view(2, 1, 1)
            image_channels = (
                image_channels - mean
            ) / std.clamp_min(1.0e-6)

        height, width = image_channels.shape[-2:]
        y_coordinates = torch.linspace(
            -1.0, 1.0, height, dtype=image_channels.dtype
        ).view(1, height, 1).expand(1, height, width)
        x_coordinates = torch.linspace(
            -1.0, 1.0, width, dtype=image_channels.dtype
        ).view(1, 1, width).expand(1, height, width)

        return torch.cat(
            [image_channels, y_coordinates, x_coordinates], dim=0
        )


class BUSBRADataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        transform: BUSBRATransform,
        flatten: bool = False,
        return_metadata: bool = False,
    ):
        self.records = [dict(record) for record in records]
        self.transform = transform
        self.flatten = bool(flatten)
        self.return_metadata = bool(return_metadata)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["image_path"]) as opened:
            image = opened.convert("L").copy()

        tensor = self.transform(image)
        if self.flatten:
            tensor = tensor.flatten()

        label = torch.tensor(int(record["label"]), dtype=torch.long)
        if not self.return_metadata:
            return tensor, label

        metadata = {
            "image_id": str(record["image_id"]),
            "case_id": str(record["case_id"]),
            "source_dataset": str(record["source_dataset"]),
            "pathology": str(record["pathology"]),
            "image_path": str(record["image_path"]),
            "fold": int(record["fold"]),
        }
        return tensor, label, metadata


@torch.no_grad()
def compute_training_channel_statistics(
    train_records: Sequence[Dict[str, object]],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    raw_dataset = BUSBRADataset(
        train_records,
        BUSBRATransform(train=False),
        flatten=False,
        return_metadata=False,
    )
    loader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    channel_sum = torch.zeros(2, dtype=torch.float64)
    channel_square_sum = torch.zeros(2, dtype=torch.float64)
    pixel_count = 0

    for tensors, _ in loader:
        values = tensors[:, :2].to(torch.float64)
        channel_sum += values.sum(dim=(0, 2, 3))
        channel_square_sum += values.square().sum(dim=(0, 2, 3))
        pixel_count += values.shape[0] * values.shape[2] * values.shape[3]

    if pixel_count == 0:
        raise RuntimeError("Cannot compute statistics from an empty dataset")

    mean = channel_sum / pixel_count
    variance = channel_square_sum / pixel_count - mean.square()
    std = variance.clamp_min(1.0e-12).sqrt()
    return tuple(mean.tolist()), tuple(std.tolist())


def get_busbra_dataloaders(
    busbra_root_dir: str | Path = "/content/drive/MyDrive/BUS/BUSBRA",
    busbra_csv_filename: str = "5-fold-cv.csv",
    test_fold: int = 1,
    batch_size: int = 8,
    evaluation_batch_size: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
    flatten: bool = False,
    return_metadata: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[str, object]]:
    """Build BUS-BRA loaders from one official test fold."""
    set_seed(seed)
    records = discover_busbra_images(
        root_dir=busbra_root_dir,
        csv_filename=busbra_csv_filename,
    )
    train_records, test_records = split_busbra_official_fold(
        records, test_fold=test_fold
    )

    channel_mean, channel_std = compute_training_channel_statistics(
        train_records,
        batch_size=max(1, batch_size),
        num_workers=num_workers,
    )
    train_transform = BUSBRATransform(
        train=True,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    test_transform = BUSBRATransform(
        train=False,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    train_dataset = BUSBRADataset(
        train_records,
        train_transform,
        flatten=flatten,
        return_metadata=return_metadata,
    )
    test_dataset = BUSBRADataset(
        test_records,
        test_transform,
        flatten=flatten,
        return_metadata=return_metadata,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    common_loader_options = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **common_loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=evaluation_batch_size or batch_size,
        shuffle=False,
        **common_loader_options,
    )

    def count(
        records_to_count: Sequence[Dict[str, object]],
        label: Optional[int] = None,
    ) -> int:
        return sum(
            label is None or int(record["label"]) == label
            for record in records_to_count
        )

    info: Dict[str, object] = {
        "dataset": "BUSBRA",
        "root_dir": str(Path(busbra_root_dir)),
        "csv_path": str(Path(busbra_root_dir) / busbra_csv_filename),
        "official_test_fold": test_fold,
        "training_folds": sorted(OFFICIAL_FOLDS.difference({test_fold})),
        "classes": dict(LABEL_TO_INDEX),
        "split_unit": "independent image",
        "views_grouped": False,
        "segmentation_masks_used": False,
        "image_size": IMAGE_SIZE,
        "output_channels": [
            "grayscale",
            "scharr_edge",
            "y_coordinate",
            "x_coordinate",
        ],
        "output_shape_before_flatten": f"4x{IMAGE_SIZE}x{IMAGE_SIZE}",
        "flattened_features": 4 * IMAGE_SIZE * IMAGE_SIZE,
        "seed": seed,
        "total_images": len(records),
        "train_images": len(train_records),
        "test_images": len(test_records),
        "total_benign": count(records, 0),
        "total_malignant": count(records, 1),
        "train_benign": count(train_records, 0),
        "train_malignant": count(train_records, 1),
        "test_benign": count(test_records, 0),
        "test_malignant": count(test_records, 1),
        "channel_mean_gray_edge": channel_mean,
        "channel_std_gray_edge": channel_std,
    }

    return train_loader, test_loader, info


def visualize_preprocessing(
    busbra_root_dir: str | Path = "/content/drive/MyDrive/BUS/BUSBRA",
    busbra_csv_filename: str = "5-fold-cv.csv",
    seed: Optional[int] = None,
    save_path: Optional[str | Path] = None,
):
    import matplotlib.pyplot as plt

    records = discover_busbra_images(
        busbra_root_dir, csv_filename=busbra_csv_filename
    )
    rng = np.random.default_rng(seed)
    record = records[int(rng.integers(0, len(records)))]

    with Image.open(record["image_path"]) as opened:
        original = opened.convert("L").copy()

    transform = BUSBRATransform(train=False)
    grayscale, enhanced, fixed_canvas, _ = transform.deterministic_stages(
        original
    )
    edge = transform.edge_extractor(fixed_canvas).squeeze(0).numpy()
    output = transform(original)

    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    displays = [
        (np.asarray(original), "Original"),
        (np.asarray(grayscale, dtype=np.float32) / 255.0, "Grayscale"),
        (np.asarray(enhanced, dtype=np.float32) / 255.0, "CLAHE"),
        (
            np.asarray(fixed_canvas, dtype=np.float32) / 255.0,
            f"Aspect resize and mean padding ({IMAGE_SIZE}x{IMAGE_SIZE})",
        ),
        (edge, "Gaussian and Scharr edge"),
        (output[1].numpy(), "Edge channel"),
        (output[2].numpy(), "Y-coordinate channel"),
        (output[3].numpy(), "X-coordinate channel"),
    ]

    for axis, (array, title) in zip(axes.flat, displays):
        axis.imshow(array, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(
        f"{record['image_id']} | {record['pathology']} | fold {record['fold']}",
        fontsize=14,
    )
    figure.tight_layout()

    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=200, bbox_inches="tight")

    plt.show()
    return figure, dict(record)


def visualize_random_augmentation(
    busbra_root_dir: str | Path = "/content/drive/MyDrive/BUS/BUSBRA",
    busbra_csv_filename: str = "5-fold-cv.csv",
    seed: Optional[int] = None,
):
    import matplotlib.pyplot as plt

    records = discover_busbra_images(
        busbra_root_dir, csv_filename=busbra_csv_filename
    )
    rng = np.random.default_rng(seed)
    record = records[int(rng.integers(0, len(records)))]

    with Image.open(record["image_path"]) as opened:
        original = opened.convert("L").copy()

    transform = BUSBRATransform(train=True)
    augmented = transform(original)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(augmented[0].numpy(), cmap="gray")
    axes[0].set_title("Augmented grayscale")
    axes[1].imshow(augmented[1].numpy(), cmap="gray")
    axes[1].set_title("Aligned augmented edge")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{record['image_id']} | {record['pathology']} | fold {record['fold']}"
    )
    figure.tight_layout()
    plt.show()
    return figure


if __name__ == "__main__":
    root = "/content/drive/MyDrive/BUS/BUSBRA"

    train_loader, test_loader, dataset_info = get_busbra_dataloaders(
        busbra_root_dir=root,
        busbra_csv_filename="5-fold-cv.csv",
        test_fold=1,
        batch_size=8,
        evaluation_batch_size=8,
        num_workers=0,
        seed=42,
        flatten=False,
        return_metadata=True,
    )

    print(dataset_info)
    images, labels, metadata = next(iter(train_loader))
    print("Training batch image shape:", tuple(images.shape))
    print("Training batch label shape:", tuple(labels.shape))
    print("First batch case IDs:", metadata["case_id"][:3])
