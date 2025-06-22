import torch
import torch.nn as nn
from jfrft import get_joint_jfrt_pair

class TrainableDiagonalFilter(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
        self.device = None
        self.filter = nn.Parameter(torch.ones(size, dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.device is None:
            self.device = x.device
            self.filter.data = self.filter.data.to(self.device)
        return torch.matmul(torch.diag(self.filter), x)

    def __repr__(self) -> str:
        return f"TrainableDiagonalFilter(size={self.size})"


class IdealLowpassFilter(nn.Module):
    def __init__(self, size: int, cutoff_count: int = None) -> None:
        super().__init__()
        self.size = size
        self.cutoff_count = cutoff_count
        pattern = [1, 1, 1, 1, 0, 0]
        repeats = (size - 12) // len(pattern)
        remainder = (size - 12) % len(pattern)
        self.filter = torch.tensor(
            pattern * repeats + pattern[:remainder] + [0] * 12, dtype=torch.float32
        )

        self.device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.device is None:
            self.device = x.device
            self.filter = self.filter.to(self.device)
        return torch.mul(self.filter.unsqueeze(0).T, x)

    def __repr__(self) -> str:
        return f"IdealLowpassFilter(size={self.size}, cutoff_count={self.cutoff_count})"

class GFRFTFilterLayer(nn.Module):
    def __init__(
        self,
        gfrft_obj,
        signal_shape: tuple,
        gfrft_order: float,
        dfrft_order: float,
        trainable_transform: bool = True,
        trainable_filter: bool = False,
        approx_order: int = 2,
        cutoff_count: int = 0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.gfrft_obj = gfrft_obj
        self.signal_shape = signal_shape
        self.flat_size = 60
        self.approx_order = approx_order
        self.device = device

        self.gfrft_order = nn.Parameter(
            torch.tensor(gfrft_order, dtype=torch.float32),
            requires_grad=trainable_transform,
        )
        self.dfrft_order = nn.Parameter(
            torch.tensor(dfrft_order, dtype=torch.float32),
            requires_grad=trainable_transform,
        )

        if trainable_filter:
            self.filter = TrainableDiagonalFilter(size=self.flat_size).to(self.device)
        else:
            self.filter = IdealLowpassFilter(self.flat_size, cutoff_count).to(self.device)

    def _update_transform_matrices(self):
        joint_jfrt_mtx, joint_ijfrt_mtx = get_joint_jfrt_pair(
            self.gfrft_obj,
            torch.zeros((6, 6), dtype=torch.complex64, device=self.device),
            self.gfrft_order,
            self.dfrft_order,
            device=self.device,
        )
        return joint_jfrt_mtx.to(torch.complex64), joint_ijfrt_mtx.to(torch.complex64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.complex64)
        parts = [x[:, i * 6:(i + 1) * 6] for i in range(6)]
        flattened_parts = [part.permute(1, 0).contiguous().view(-1, 1) for part in parts]
        combined = torch.cat(flattened_parts, dim=1)
        self.joint_jfrt_mtx, self.joint_ijfrt_mtx = self._update_transform_matrices()
        transformed = torch.matmul(self.joint_jfrt_mtx, combined)
        filtered = self.filter(transformed)
        restored = torch.matmul(self.joint_ijfrt_mtx, filtered)
        restored_parts = [
            restored[:, i].view(6, 10).permute(1, 0) for i in range(6)
        ]
        out = torch.cat(restored_parts, dim=1)
        return out

    def get_filter_coefficients(self):
        if isinstance(self.filter, TrainableDiagonalFilter):
            return self.filter.filter.data.cpu().numpy()
        elif isinstance(self.filter, IdealLowpassFilter):
            return self.filter.filter.cpu().numpy()
        return None
