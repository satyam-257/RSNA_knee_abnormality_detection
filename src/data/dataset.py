import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .dicom_utils import load_dicom_slice, sample_uniform_slices, normalise_laterality, order_slices_3d, infer_laterality_from_geometry
from src.config import SLOTS, N_SLOT, LABELS

class KneeMRISlotDataset(Dataset):
    """
    6-Slot Multi-View PyTorch Dataset for Knee MRI Studies.
    Organizes DICOM series according to the 6 clinical slots:
        Slot 0: Sagittal Fluid-Sensitive (1)
        Slot 1: Coronal Fluid-Sensitive (1)
        Slot 2: Axial Fluid-Sensitive (1)
        Slot 3: Sagittal Non-Fluid (0)
        Slot 4: Coronal Non-Fluid (0)
        Slot 5: Axial Non-Fluid (0)
    Applies laterality normalization (standardizing left vs right knees)
    and true 3D spatial vector ordering.
    """
    def __init__(
        self,
        df_studies: pd.DataFrame,
        df_series: pd.DataFrame,
        data_dir: Path,
        labels: Optional[List[str]] = None,
        num_slices: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        crop_mm: float = 135.0,
        is_training: bool = True
    ):
        self.df_studies = df_studies.reset_index(drop=True)
        self.df_series = df_series
        self.data_dir = Path(data_dir)
        self.labels = labels if labels is not None else LABELS
        self.num_slices = num_slices
        self.target_size = target_size
        self.crop_mm = crop_mm
        self.is_training = is_training

        # Pre-group series by StudyInstanceUID for fast lookup
        self.series_by_study = {}
        for study_id, group in self.df_series.groupby("StudyInstanceUID"):
            self.series_by_study[study_id] = group

    def __len__(self) -> int:
        return len(self.df_studies)

    def _load_slot_series(self, study_id: str, plane: str, fluid_sensitive: int, laterality: Optional[str] = None) -> Tuple[torch.Tensor, float, Optional[str]]:
        """
        Loads slices for a specific (plane, fluid_sensitive) slot.
        Returns:
            tensor: (K, 1, H, W)
            mask_val: 1.0 if series exists and loaded, 0.0 otherwise.
            inferred_lat: 'L', 'R', or None.
        """
        K = self.num_slices
        H, W = self.target_size

        if study_id not in self.series_by_study:
            return torch.zeros((K, 1, H, W), dtype=torch.float32), 0.0, laterality

        study_series = self.series_by_study[study_id]
        match = study_series[
            (study_series["Anatomical_Plane"].str.lower() == plane.lower()) &
            (study_series["Fluid_Sensitive"] == fluid_sensitive)
        ]

        # If exact match not found, fallback to plane match if available
        if len(match) == 0:
            match = study_series[study_series["Anatomical_Plane"].str.lower() == plane.lower()]
            if len(match) == 0:
                return torch.zeros((K, 1, H, W), dtype=torch.float32), 0.0, laterality

        selected_row = match.iloc[0]
        series_id = str(selected_row["SeriesInstanceUID"])

        possible_dirs = [
            self.data_dir / "train_series" / study_id / series_id,
            self.data_dir / "test_series" / study_id / series_id,
            self.data_dir / study_id / series_id
        ]

        series_path = None
        for p in possible_dirs:
            if p.exists() and p.is_dir():
                series_path = p
                break

        if series_path is None:
            return torch.zeros((K, 1, H, W), dtype=torch.float32), 0.0, laterality

        dcm_files = list(series_path.glob("*.dcm"))
        if len(dcm_files) == 0:
            return torch.zeros((K, 1, H, W), dtype=torch.float32), 0.0, laterality

        # Physical 3D ordering along volume normal
        ordered_files = order_slices_3d(dcm_files)

        # Infer laterality geometrically if not established
        if not laterality and len(ordered_files) > 0:
            laterality = infer_laterality_from_geometry(ordered_files[len(ordered_files) // 2])

        sampled_paths = sample_uniform_slices(ordered_files, num_slices=K)

        slices_arr = []
        for path in sampled_paths:
            try:
                arr = load_dicom_slice(path, target_size=self.target_size, crop_mm=self.crop_mm)
            except Exception:
                arr = np.zeros((H, W), dtype=np.float32)
            slices_arr.append(arr)

        tensor = torch.from_numpy(np.stack(slices_arr, axis=0)).unsqueeze(1) # (K, 1, H, W)

        # Apply laterality normalization (flip right knee)
        tensor = normalise_laterality(tensor, plane=plane, lat=laterality)

        return tensor, 1.0, laterality

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        row = self.df_studies.iloc[idx]
        study_id = str(row["StudyInstanceUID"])
        laterality = getattr(row, "Laterality", None)
        if pd.isna(laterality) or not laterality:
            report_text = str(row.get("Report", "")).lower()
            if "right" in report_text and "left" not in report_text:
                laterality = "R"
            elif "left" in report_text and "right" not in report_text:
                laterality = "L"
            else:
                laterality = None

        slot_tensors = []
        slot_masks = []

        for slot_idx, (plane, fluid) in enumerate(SLOTS):
            tensor, mask_val, inferred_lat = self._load_slot_series(study_id, plane, fluid, laterality)
            if not laterality and inferred_lat:
                laterality = inferred_lat
            slot_tensors.append(tensor)
            slot_masks.append(mask_val)

        slot_tensors = torch.stack(slot_tensors, dim=0) # (N_SLOT, K, 1, H, W)
        slot_mask = torch.tensor(slot_masks, dtype=torch.float32) # (N_SLOT,)

        sample = {
            "study_id": study_id,
            "slot_tensors": slot_tensors,
            "slot_mask": slot_mask
        }

        if self.is_training:
            label_vals = []
            for col in self.labels:
                col_name = col + "_pseudo" if (col + "_pseudo") in row else col
                val = float(row[col_name]) if pd.notnull(row[col_name]) else 0.0
                label_vals.append(val)
            sample["targets"] = torch.tensor(label_vals, dtype=torch.float32)

        return sample
