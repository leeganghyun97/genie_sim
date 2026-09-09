"""CSV, TensorBoard, and optional Weights & Biases logging for SAC runners.

The logger deliberately has no simulator or training-framework dependency.  It
writes scalar metrics in a canonical long CSV format and mirrors each scalar to
TensorBoard using the same metric name supplied by the caller.  Transition and
episode schemas are established by their first row and then held fixed,
including when a run is resumed with ``append=True``.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from numbers import Integral, Real
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Sequence, TextIO


_METRIC_FIELDS = ("timestamp", "event", "global_step", "metric", "value")


class Stage2RunLogger:
    """Write Stage 2 scalar, transition, and episode artifacts.

    Args:
        output_dir: Directory that receives the CSV files and TensorBoard log.
        tensorboard_enabled: Mirror scalar metrics to TensorBoard when true.
        append: Append to existing CSV files while preserving their headers.
        writer_factory: Optional ``SummaryWriter``-compatible factory.  This is
            intended for dependency-free tests; production callers normally
            leave it unset.
        wandb_enabled: Mirror scalar metrics to W&B when explicitly enabled.
        wandb_project/entity/run_name/mode: W&B run identity and transport mode.
        wandb_group/job_type/tags: Optional W&B comparison organization metadata.
        comparison_id: Stable experiment-comparison identifier stored in config.
        wandb_config: Immutable run metadata copied into the W&B run config.
        wandb_init: Optional injectable ``wandb.init`` replacement for tests.

    ``transitions.csv`` and ``episodes.csv`` accept flat mappings.  Their first
    row defines the header.  Later rows must contain exactly the same keys,
    although mapping order may differ.
    """

    def __init__(
        self,
        output_dir: str | Path,
        tensorboard_enabled: bool = True,
        append: bool = False,
        *,
        writer_factory: Callable[[str], Any] | None = None,
        wandb_enabled: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        wandb_mode: str = "online",
        wandb_group: str | None = None,
        wandb_job_type: str | None = None,
        wandb_tags: Sequence[str] | None = None,
        comparison_id: str | None = None,
        wandb_config: Mapping[str, Any] | None = None,
        wandb_init: Callable[..., Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_enabled = bool(tensorboard_enabled)
        self.append = bool(append)
        self._lock = threading.RLock()
        self._closed = False
        self._tensorboard_writer: Any | None = None
        self.wandb_enabled = bool(wandb_enabled)
        self._wandb_run: Any | None = None
        self._wandb_run_identity: dict[str, Any] | None = None
        self._streams: list[TextIO] = []

        if self.wandb_enabled:
            if not isinstance(wandb_project, str) or not wandb_project.strip():
                raise ValueError(
                    "wandb_project must be a non-empty string when W&B is enabled"
                )
            if wandb_mode not in {"online", "offline"}:
                raise ValueError("wandb_mode must be 'online' or 'offline'")
            self._wandb_run = self._create_wandb_run(
                project=wandb_project.strip(),
                entity=wandb_entity,
                run_name=wandb_run_name,
                mode=wandb_mode,
                group=wandb_group,
                job_type=wandb_job_type,
                tags=wandb_tags,
                comparison_id=comparison_id,
                config=wandb_config,
                init_function=wandb_init,
            )
            self._wandb_run_identity = self._extract_wandb_run_identity(
                self._wandb_run,
                requested_project=wandb_project.strip(),
                requested_entity=wandb_entity,
                requested_run_name=wandb_run_name,
                mode=wandb_mode,
                requested_group=wandb_group,
                requested_job_type=wandb_job_type,
                requested_tags=wandb_tags,
                comparison_id=comparison_id,
            )

        # Resolve TensorBoard before opening CSV files so an unavailable
        # requested dependency cannot leave a partially initialized run.
        if self.tensorboard_enabled:
            self._tensorboard_writer = self._create_summary_writer(
                self.output_dir / "tensorboard",
                writer_factory=writer_factory,
            )

        try:
            self._metrics_stream, self._metrics_writer = self._open_metrics_csv()
            (
                self._transition_stream,
                self._transition_writer,
                self._transition_fields,
            ) = self._open_dynamic_csv("transitions.csv")
            (
                self._episode_stream,
                self._episode_writer,
                self._episode_fields,
            ) = self._open_dynamic_csv("episodes.csv")
        except Exception:
            self._close_resources()
            raise

    def _create_wandb_run(
        self,
        *,
        project: str,
        entity: str | None,
        run_name: str | None,
        mode: str,
        group: str | None,
        job_type: str | None,
        tags: Sequence[str] | None,
        comparison_id: str | None,
        config: Mapping[str, Any] | None,
        init_function: Callable[..., Any] | None,
    ) -> Any:
        if init_function is None:
            try:
                import wandb
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "W&B logging was requested, but the wandb package is "
                    "unavailable. Install wandb or run with --no-wandb."
                ) from error
            init_function = wandb.init

        normalized_group = self._optional_wandb_text("wandb_group", group)
        normalized_job_type = self._optional_wandb_text(
            "wandb_job_type", job_type
        )
        normalized_tags = self._wandb_tags(tags)
        normalized_comparison_id = self._optional_wandb_text(
            "comparison_id", comparison_id
        )
        run_config = dict(config or {})
        comparison_metadata: dict[str, Any] = {
            "wandb_group": normalized_group,
            "wandb_job_type": normalized_job_type,
            "wandb_tags": list(normalized_tags),
            "comparison_id": normalized_comparison_id,
        }
        for key, value in comparison_metadata.items():
            if value not in (None, []):
                run_config[key] = value

        kwargs: dict[str, Any] = {
            "project": project,
            "dir": str(self.output_dir),
            "mode": mode,
            "config": run_config,
            # Boolean ``reinit`` is deprecated by current W&B releases.  The
            # explicit policy retains the historical behavior: finish any
            # previous run before creating this runner-owned run.
            "reinit": "finish_previous",
        }
        if entity is not None:
            if not isinstance(entity, str) or not entity.strip():
                raise ValueError("wandb_entity must be a non-empty string")
            kwargs["entity"] = entity.strip()
        if run_name is not None:
            if not isinstance(run_name, str) or not run_name.strip():
                raise ValueError("wandb_run_name must be a non-empty string")
            kwargs["name"] = run_name.strip()
        if normalized_group is not None:
            kwargs["group"] = normalized_group
        if normalized_job_type is not None:
            kwargs["job_type"] = normalized_job_type
        if normalized_tags:
            kwargs["tags"] = list(normalized_tags)
        try:
            run = init_function(**kwargs)
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize W&B project {project!r} in {mode!r} mode"
            ) from error
        if run is None or not callable(getattr(run, "log", None)):
            raise RuntimeError("W&B initialization did not return a usable run")
        # W&B's implicit internal step advances after every committed ``log``
        # call.  A training transition, optimization update, and completed
        # episode can legitimately share one environment ``global_step``; using
        # W&B's ``step=`` argument for each call therefore makes the later call
        # look out-of-order.  Keep W&B's write index monotonic and use the
        # explicit environment transition count as the chart x-axis instead.
        define_metric = getattr(run, "define_metric", None)
        if callable(define_metric):
            define_metric("global_step")
            define_metric("*", step_metric="global_step")
        return run

    @staticmethod
    def _optional_wandb_text(name: str, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string when supplied")
        return value.strip()

    @classmethod
    def _wandb_tags(cls, values: Sequence[str] | None) -> tuple[str, ...]:
        if values is None:
            return ()
        if isinstance(values, (str, bytes)):
            raise ValueError("wandb_tags must be a sequence of tag strings")
        normalized: list[str] = []
        for value in values:
            tag = cls._optional_wandb_text("wandb_tags item", value)
            assert tag is not None
            if tag not in normalized:
                normalized.append(tag)
        return tuple(normalized)

    @staticmethod
    def _safe_wandb_text(value: Any) -> str | None:
        """Return one JSON-safe W&B identity value without coercing objects."""

        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    @classmethod
    def _wandb_attribute(cls, run: Any, name: str) -> str | None:
        """Read a W&B run attribute while treating SDK properties as untrusted."""

        try:
            value = getattr(run, name, None)
        except Exception:
            return None
        return cls._safe_wandb_text(value)

    @classmethod
    def _extract_wandb_run_identity(
        cls,
        run: Any,
        *,
        requested_project: str,
        requested_entity: str | None,
        requested_run_name: str | None,
        mode: str,
        requested_group: str | None,
        requested_job_type: str | None,
        requested_tags: Sequence[str] | None,
        comparison_id: str | None,
    ) -> dict[str, Any]:
        """Capture the stable, non-secret subset of a successfully opened run."""

        project = cls._wandb_attribute(run, "project") or requested_project
        entity = cls._wandb_attribute(run, "entity") or cls._safe_wandb_text(
            requested_entity
        )
        name = cls._wandb_attribute(run, "name") or cls._safe_wandb_text(
            requested_run_name
        )
        url = cls._wandb_attribute(run, "url") if mode == "online" else None
        if url is None and mode == "online":
            try:
                get_url = getattr(run, "get_url", None)
            except Exception:
                get_url = None
            if callable(get_url):
                try:
                    url = cls._safe_wandb_text(get_url())
                except Exception:
                    url = None
        return {
            "id": cls._wandb_attribute(run, "id"),
            "name": name,
            "project": project,
            "entity": entity,
            "url": url,
            "mode": mode,
            "group": cls._optional_wandb_text("wandb_group", requested_group),
            "job_type": cls._optional_wandb_text(
                "wandb_job_type", requested_job_type
            ),
            "tags": list(cls._wandb_tags(requested_tags)),
            "comparison_id": cls._optional_wandb_text(
                "comparison_id", comparison_id
            ),
        }

    @property
    def wandb_run_identity(self) -> dict[str, Any] | None:
        """Return a JSON-serializable copy of the initialized W&B identity.

        The evidence remains available after :meth:`close`; no-W&B loggers
        return ``None``.  Returning a copy prevents callers from mutating the
        identity that will be persisted in the run manifest.
        """

        if self._wandb_run_identity is None:
            return None
        identity = dict(self._wandb_run_identity)
        identity["tags"] = list(identity["tags"])
        return identity

    @staticmethod
    def _create_summary_writer(
        log_dir: Path,
        *,
        writer_factory: Callable[[str], Any] | None,
    ) -> Any:
        if writer_factory is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "TensorBoard logging was requested, but "
                    "torch.utils.tensorboard.SummaryWriter is unavailable. "
                    "Install the tensorboard dependency or explicitly set "
                    "tensorboard_enabled=False."
                ) from error
            writer_factory = SummaryWriter

        try:
            return writer_factory(str(log_dir))
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize TensorBoard SummaryWriter at {log_dir}"
            ) from error

    def _open_metrics_csv(self) -> tuple[TextIO, csv.DictWriter]:
        path = self.output_dir / "metrics.csv"
        if self.append:
            existing_fields = self._read_existing_header(path)
            if existing_fields is not None and tuple(existing_fields) != _METRIC_FIELDS:
                raise ValueError(
                    f"Existing metrics.csv header {existing_fields!r} does not "
                    f"match the canonical header {list(_METRIC_FIELDS)!r}"
                )

        stream = path.open("a" if self.append else "w", encoding="utf-8", newline="")
        self._streams.append(stream)
        writer = csv.DictWriter(stream, fieldnames=list(_METRIC_FIELDS))
        if stream.tell() == 0:
            writer.writeheader()
            stream.flush()
        return stream, writer

    def _open_dynamic_csv(
        self, filename: str
    ) -> tuple[TextIO, csv.DictWriter | None, tuple[str, ...] | None]:
        path = self.output_dir / filename
        existing_fields = self._read_existing_header(path) if self.append else None
        stream = path.open("a" if self.append else "w", encoding="utf-8", newline="")
        self._streams.append(stream)
        if existing_fields is None:
            return stream, None, None
        fields = self._validate_header(filename, existing_fields)
        return stream, csv.DictWriter(stream, fieldnames=list(fields)), fields

    @staticmethod
    def _read_existing_header(path: Path) -> list[str] | None:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        with path.open("r", encoding="utf-8", newline="") as stream:
            try:
                return next(csv.reader(stream))
            except StopIteration:
                return None

    @staticmethod
    def _validate_header(filename: str, fields: list[str]) -> tuple[str, ...]:
        if not fields or any(not field for field in fields):
            raise ValueError(f"Existing {filename} has an empty CSV header")
        if len(fields) != len(set(fields)):
            raise ValueError(f"Existing {filename} has duplicate CSV header fields")
        return tuple(fields)

    @staticmethod
    def _timestamp() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _global_step(value: Any) -> int:
        scalar = Stage2RunLogger._python_scalar("global_step", value)
        if isinstance(scalar, bool) or not isinstance(scalar, Integral):
            raise TypeError("global_step must be an integer")
        step = int(scalar)
        if step < 0:
            raise ValueError("global_step cannot be negative")
        return step

    @staticmethod
    def _python_scalar(name: str, value: Any) -> Any:
        """Convert NumPy/Torch-style scalar objects without importing either."""

        if isinstance(value, (str, bytes, bool, Integral, Real)) or value is None:
            return value
        item = getattr(value, "item", None)
        if callable(item):
            try:
                scalar = item()
            except (TypeError, ValueError) as error:
                raise TypeError(f"{name} must be a scalar value") from error
            if scalar is value:
                raise TypeError(f"{name} must be a scalar value")
            return Stage2RunLogger._python_scalar(name, scalar)
        return value

    @classmethod
    def _finite_number(cls, name: str, value: Any) -> float:
        scalar = cls._python_scalar(name, value)
        if isinstance(scalar, bool):
            return float(int(scalar))
        if not isinstance(scalar, Real):
            raise TypeError(f"{name} must be a real numeric scalar")
        numeric = float(scalar)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite, got {scalar!r}")
        return numeric

    @classmethod
    def _validated_row(cls, artifact: str, row: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise TypeError(f"{artifact} row must be a mapping")
        if not row:
            raise ValueError(f"{artifact} row cannot be empty")
        validated: dict[str, Any] = {}
        for key, value in row.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{artifact} row keys must be non-empty strings")
            scalar = cls._python_scalar(f"{artifact}.{key}", value)
            if isinstance(scalar, bool):
                validated[key] = scalar
            elif isinstance(scalar, Real):
                if not math.isfinite(float(scalar)):
                    raise ValueError(
                        f"{artifact}.{key} must be finite, got {scalar!r}"
                    )
                validated[key] = scalar
            elif scalar is None or isinstance(scalar, (str, bytes)):
                validated[key] = scalar
            else:
                raise TypeError(
                    f"{artifact}.{key} must be a flat CSV scalar, "
                    f"got {type(scalar).__name__}"
                )
        return validated

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Stage2RunLogger is closed")

    def log_scalars(
        self,
        event: str,
        global_step: int,
        metrics: Mapping[str, Any],
    ) -> None:
        """Write finite scalar metrics to CSV and TensorBoard."""

        with self._lock:
            self._ensure_open()
            if not isinstance(event, str) or not event.strip():
                raise ValueError("event must be a non-empty string")
            step = self._global_step(global_step)
            if not isinstance(metrics, Mapping):
                raise TypeError("metrics must be a mapping")

            validated: list[tuple[str, float]] = []
            for metric, value in metrics.items():
                if not isinstance(metric, str) or not metric:
                    raise ValueError("metric names must be non-empty strings")
                validated.append(
                    (metric, self._finite_number(f"metrics.{metric}", value))
                )

            timestamp = self._timestamp()
            for metric, value in validated:
                self._metrics_writer.writerow(
                    {
                        "timestamp": timestamp,
                        "event": event,
                        "global_step": step,
                        "metric": metric,
                        "value": value,
                    }
                )
            self._metrics_stream.flush()

            if self._tensorboard_writer is not None:
                for metric, value in validated:
                    self._tensorboard_writer.add_scalar(metric, value, step)
                flush = getattr(self._tensorboard_writer, "flush", None)
                if callable(flush):
                    flush()

            if self._wandb_run is not None:
                self._wandb_run.log(
                    {
                        "global_step": step,
                        **{metric: value for metric, value in validated},
                    },
                    commit=True,
                )

    def _log_dynamic_row(self, artifact: str, row: Mapping[str, Any]) -> None:
        with self._lock:
            self._ensure_open()
            validated = self._validated_row(artifact, row)
            if artifact == "transition":
                stream = self._transition_stream
                writer = self._transition_writer
                fields = self._transition_fields
            else:
                stream = self._episode_stream
                writer = self._episode_writer
                fields = self._episode_fields

            if fields is None:
                fields = tuple(validated.keys())
                writer = csv.DictWriter(stream, fieldnames=list(fields))
                writer.writeheader()
                if artifact == "transition":
                    self._transition_fields = fields
                    self._transition_writer = writer
                else:
                    self._episode_fields = fields
                    self._episode_writer = writer
            elif set(validated) != set(fields):
                missing = sorted(set(fields) - set(validated))
                extra = sorted(set(validated) - set(fields))
                raise ValueError(
                    f"{artifact} row schema does not match its stable header; "
                    f"missing={missing}, extra={extra}"
                )

            assert writer is not None
            writer.writerow(validated)
            stream.flush()

    def log_transition(self, row: Mapping[str, Any]) -> None:
        """Append one finite, flat transition row and flush it immediately."""

        self._log_dynamic_row("transition", row)

    def log_episode(self, row: Mapping[str, Any]) -> None:
        """Append one finite, flat episode row and flush it immediately."""

        self._log_dynamic_row("episode", row)

    def _close_resources(self) -> None:
        wandb_run = self._wandb_run
        self._wandb_run = None
        if wandb_run is not None:
            finish = getattr(wandb_run, "finish", None)
            if callable(finish):
                finish()
        writer = self._tensorboard_writer
        self._tensorboard_writer = None
        if writer is not None:
            flush = getattr(writer, "flush", None)
            if callable(flush):
                flush()
            close = getattr(writer, "close", None)
            if callable(close):
                close()
        for stream in self._streams:
            if not stream.closed:
                stream.flush()
                stream.close()
        self._streams.clear()

    def close(self) -> None:
        """Flush and close every output.  Calling twice is safe."""

        with self._lock:
            if self._closed:
                return
            self._close_resources()
            self._closed = True

    def __enter__(self) -> "Stage2RunLogger":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
