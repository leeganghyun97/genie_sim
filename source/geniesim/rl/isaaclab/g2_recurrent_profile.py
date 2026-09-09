"""Named recurrent hyperparameter profiles shared by G2 training entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


G2_RECURRENT_PROFILE_FILE = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "model"
    / "g2_recurrent_profiles.json"
)


@dataclass(frozen=True)
class G2RecurrentProfile:
    name: str
    hidden_dim: int
    gru_num_layers: int
    sequence_length: int
    burn_in_steps: int
    sequence_stride: int
    gradient_clip: float

    def validated(self) -> "G2RecurrentProfile":
        if self.hidden_dim <= 0:
            raise ValueError("recurrent profile hidden_dim must be positive")
        if self.gru_num_layers != 1:
            raise ValueError("G2 recurrent profiles require exactly one GRU layer")
        if not 0 <= self.burn_in_steps < self.sequence_length:
            raise ValueError("recurrent profile burn-in must be inside the sequence")
        if not 0 < self.sequence_stride <= self.sequence_length:
            raise ValueError("recurrent profile stride must be inside the sequence")
        if self.gradient_clip <= 0.0:
            raise ValueError("recurrent profile gradient clip must be positive")
        return self

    def serializable(self) -> dict[str, int | float | str]:
        return {
            "name": self.name,
            "hidden_dim": self.hidden_dim,
            "gru_num_layers": self.gru_num_layers,
            "sequence_length": self.sequence_length,
            "burn_in_steps": self.burn_in_steps,
            "sequence_stride": self.sequence_stride,
            "gradient_clip": self.gradient_clip,
        }


def _profile_document(path: str | Path = G2_RECURRENT_PROFILE_FILE) -> dict:
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schema") != "geniesim_g2_recurrent_profiles_v1":
        raise ValueError("G2 recurrent profile schema differs")
    if not isinstance(document.get("profiles"), dict):
        raise ValueError("G2 recurrent profile document has no profiles")
    return document


def recurrent_profile_names(path: str | Path = G2_RECURRENT_PROFILE_FILE) -> tuple[str, ...]:
    return tuple(sorted(_profile_document(path)["profiles"]))


def default_recurrent_profile_name(path: str | Path = G2_RECURRENT_PROFILE_FILE) -> str:
    document = _profile_document(path)
    name = document.get("default_profile")
    if name not in document["profiles"]:
        raise ValueError("default recurrent profile is unavailable")
    return str(name)


def load_g2_recurrent_profile(
    name: str | None = None,
    path: str | Path = G2_RECURRENT_PROFILE_FILE,
) -> G2RecurrentProfile:
    document = _profile_document(path)
    selected = str(name or document["default_profile"])
    try:
        values = document["profiles"][selected]
    except KeyError as exc:
        raise ValueError(f"unknown G2 recurrent profile: {selected}") from exc
    return G2RecurrentProfile(name=selected, **values).validated()
