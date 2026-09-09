"""Transport-neutral, fail-closed remote keyboard command adapter.

The legacy Pico/ROS teleoperation stack emits absolute joint commands and is
not action-compatible with the Isaac Lab teacher/student pipeline.  A remote
keyboard frontend can instead submit this explicit physical 8-D packet; this
adapter applies the same normalization exactly once as the local keyboard.
Network transport is intentionally outside this safety contract.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from .g2_teleop_dataset import G2RedundancyKeyboardTeleopContract


G2_REMOTE_KEYBOARD_PACKET_SCHEMA = "g2_remote_keyboard_rotvec_physical_v2"
G2_REMOTE_KEYBOARD_ACTION_ORDER = (
    "dx_m", "dy_m", "dz_m", "drotvec_x_rad", "drotvec_y_rad", "drotvec_z_rad",
    "elbow_nullspace_normalized", "gripper_binary",
)
G2_REMOTE_EPISODE_COMMANDS = frozenset(
    {"CONTINUE", "SAVE", "SAVE_FAILURE", "DISCARD", "EMERGENCY_STOP"}
)
G2_REMOTE_OPERATOR_VIEWS = frozenset({"right_wrist", "head", "perspective"})
G2_REMOTE_COMMAND_FRAMES = frozenset({"robot_root", "operator_view"})


class G2RemoteCommandError(RuntimeError):
    """Base class for a rejected or unavailable remote command."""


class G2RemoteCommandStale(G2RemoteCommandError):
    """Raised instead of silently replaying a stale remote command."""


@dataclass(frozen=True)
class G2RemoteKeyboardPacket:
    sequence_id: int
    source_timestamp_s: float
    physical_action: tuple[float, float, float, float, float, float, float, float]
    schema: str = G2_REMOTE_KEYBOARD_PACKET_SCHEMA
    # Control-plane episode commands do not change the physical/action schema.
    # They are edge-triggered once per accepted sequence id by the adapter.
    episode_command: str = "CONTINUE"
    # Explicit operator stop for queued delta requests.  A neutral heartbeat
    # cannot carry this meaning because it is emitted continuously.
    clear_motion_queue: bool = False
    # Edge-triggered operator intent. Heartbeats carry the absolute state for
    # observability but cannot re-apply a close latch across an episode reset.
    gripper_command_changed: bool = False
    # This changes only the human operator's Kit viewport.  It never changes
    # which synchronized RGB-D streams are recorded or encoded.
    operator_view: str = "right_wrist"
    # Terminal W/S/A/D/Q/E axes are expressed in the selected human view.
    # Other producers remain backward-compatible with robot-root commands.
    command_frame: str = "robot_root"


class G2RemoteKeyboardCommandAdapter:
    """Validate ordering/heartbeat and return canonical local-keyboard tensors."""

    def __init__(self, *, maximum_staleness_s: float = 0.25) -> None:
        if not math.isfinite(maximum_staleness_s) or maximum_staleness_s <= 0.0:
            raise ValueError("maximum_staleness_s must be finite and positive")
        self.maximum_staleness_s = float(maximum_staleness_s)
        self.normalization = G2RedundancyKeyboardTeleopContract().validate()
        self._last_sequence_id: int | None = None
        self._last_source_timestamp_s: float | None = None
        self._last_received_monotonic_s: float | None = None
        self._physical: torch.Tensor | None = None
        self._normalized: torch.Tensor | None = None

    def submit(
        self,
        packet: G2RemoteKeyboardPacket,
        *,
        received_monotonic_s: float,
        device: str | torch.device = "cpu",
    ) -> None:
        if packet.schema != G2_REMOTE_KEYBOARD_PACKET_SCHEMA:
            raise G2RemoteCommandError("REMOTE_COMMAND_SCHEMA_MISMATCH")
        if packet.sequence_id < 0:
            raise G2RemoteCommandError("REMOTE_COMMAND_NEGATIVE_SEQUENCE")
        if self._last_sequence_id is not None and packet.sequence_id <= self._last_sequence_id:
            raise G2RemoteCommandError("REMOTE_COMMAND_REPLAY_OR_REORDER")
        if not math.isfinite(packet.source_timestamp_s):
            raise G2RemoteCommandError("REMOTE_COMMAND_SOURCE_TIMESTAMP_NONFINITE")
        if (
            self._last_source_timestamp_s is not None
            and packet.source_timestamp_s <= self._last_source_timestamp_s
        ):
            raise G2RemoteCommandError("REMOTE_COMMAND_SOURCE_TIMESTAMP_NONMONOTONIC")
        if not math.isfinite(received_monotonic_s):
            raise G2RemoteCommandError("REMOTE_COMMAND_RECEIVE_TIMESTAMP_NONFINITE")
        episode_command = str(packet.episode_command).upper()
        if episode_command not in G2_REMOTE_EPISODE_COMMANDS:
            raise G2RemoteCommandError("REMOTE_COMMAND_EPISODE_COMMAND_INVALID")
        operator_view = str(packet.operator_view).lower()
        if operator_view not in G2_REMOTE_OPERATOR_VIEWS:
            raise G2RemoteCommandError("REMOTE_COMMAND_OPERATOR_VIEW_INVALID")
        command_frame = str(packet.command_frame).lower()
        if command_frame not in G2_REMOTE_COMMAND_FRAMES:
            raise G2RemoteCommandError("REMOTE_COMMAND_FRAME_INVALID")
        physical = torch.as_tensor(
            packet.physical_action, dtype=torch.float32, device=device
        )
        if physical.shape != (8,) or not bool(torch.isfinite(physical).all()):
            raise G2RemoteCommandError("REMOTE_COMMAND_ACTION_INVALID")
        normalized = self.normalization.normalize(physical.unsqueeze(0)).squeeze(0)
        self._last_sequence_id = int(packet.sequence_id)
        self._last_source_timestamp_s = float(packet.source_timestamp_s)
        self._last_received_monotonic_s = float(received_monotonic_s)
        self._physical = physical.clone()
        self._normalized = normalized.clone()
        self._operator_view = operator_view
        self._command_frame = command_frame
        # A terminal heartbeat can arrive immediately after an edge-triggered
        # SAVE/DISCARD packet. ``G2RemoteJsonlCommandSource`` drains all newly
        # appended rows before returning one sample, so a trailing CONTINUE
        # must not erase an unconsumed operator decision.
        if episode_command != "CONTINUE":
            self._episode_command = episode_command
            self._episode_command_pending = True
        elif not getattr(self, "_episode_command_pending", False):
            self._episode_command = "CONTINUE"
            self._episode_command_pending = False

    def sample(
        self, *, now_monotonic_s: float
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
        if not math.isfinite(now_monotonic_s):
            raise G2RemoteCommandError("REMOTE_COMMAND_SAMPLE_TIMESTAMP_NONFINITE")
        if self._last_received_monotonic_s is None or self._physical is None:
            raise G2RemoteCommandStale("REMOTE_COMMAND_NOT_RECEIVED")
        age = float(now_monotonic_s) - self._last_received_monotonic_s
        if age < 0.0:
            raise G2RemoteCommandError("REMOTE_MONOTONIC_CLOCK_REVERSED")
        if age > self.maximum_staleness_s:
            raise G2RemoteCommandStale("REMOTE_COMMAND_STALE")
        episode_command_pending = bool(self._episode_command_pending)
        self._episode_command_pending = False
        return self._physical.clone(), self._normalized.clone(), {
            "schema": G2_REMOTE_KEYBOARD_PACKET_SCHEMA,
            "sequence_id": int(self._last_sequence_id),
            "receive_age_s": age,
            "source_timestamp_s": float(self._last_source_timestamp_s),
            "episode_command": self._episode_command,
            "episode_command_pending": int(episode_command_pending),
            "operator_view": self._operator_view,
            "command_frame": self._command_frame,
        }


class G2RemoteJsonlCommandSource:
    """Non-blocking append-only JSONL frontend for the remote adapter.

    The transport that writes the shared file is intentionally outside the
    robot process.  Every line is one complete packet and must be flushed by
    its writer.  EOF never blocks; the adapter heartbeat rejects a stale last
    command instead of replaying it indefinitely.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        maximum_staleness_s: float = 0.25,
    ) -> None:
        self.path = Path(path)
        self._stream = self.path.open("r", encoding="utf-8")
        self.adapter = G2RemoteKeyboardCommandAdapter(
            maximum_staleness_s=maximum_staleness_s
        )
        # Validate every wire packet in arrival order, independently of the
        # policy-rate delivery adapter.  RGB-D rendering can be much slower
        # than the terminal heartbeat, so draining directly into one adapter
        # used to overwrite a key pulse with the following neutral heartbeat.
        self._ingest_validator = G2RemoteKeyboardCommandAdapter(
            maximum_staleness_s=maximum_staleness_s
        )
        self._pending_action_packets: deque[G2RemoteKeyboardPacket] = deque()
        self._latest_packet: G2RemoteKeyboardPacket | None = None
        self._last_delivered_sequence_id: int | None = None
        self._last_delivery_was_delta = False
        self._maximum_pending_action_packets = 256

    @staticmethod
    def _packet_from_payload(payload: object) -> G2RemoteKeyboardPacket:
        if not isinstance(payload, dict):
            raise G2RemoteCommandError("REMOTE_COMMAND_JSON_NOT_OBJECT")
        clear_motion_queue = payload.get("clear_motion_queue", False)
        if not isinstance(clear_motion_queue, bool):
            raise G2RemoteCommandError("REMOTE_COMMAND_CLEAR_QUEUE_INVALID")
        gripper_command_changed = payload.get("gripper_command_changed", False)
        if not isinstance(gripper_command_changed, bool):
            raise G2RemoteCommandError("REMOTE_COMMAND_GRIPPER_EDGE_INVALID")
        try:
            packet = G2RemoteKeyboardPacket(
                schema=str(payload["schema"]),
                sequence_id=int(payload["sequence_id"]),
                source_timestamp_s=float(payload["source_timestamp_s"]),
                physical_action=tuple(float(value) for value in payload["physical_action"]),
                episode_command=str(payload.get("episode_command", "CONTINUE")),
                clear_motion_queue=clear_motion_queue,
                gripper_command_changed=gripper_command_changed,
                operator_view=str(payload.get("operator_view", "right_wrist")),
                command_frame=str(payload.get("command_frame", "robot_root")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise G2RemoteCommandError("REMOTE_COMMAND_JSON_INVALID") from exc
        if len(packet.physical_action) != 8:
            raise G2RemoteCommandError("REMOTE_COMMAND_ACTION_ORDER_INVALID")
        return packet

    @staticmethod
    def _requires_fifo_delivery(packet: G2RemoteKeyboardPacket) -> bool:
        # Translation/rotation/elbow values are delta requests and must be
        # consumed once.  Gripper is an absolute binary state and can safely
        # use the latest heartbeat value.
        motion = any(abs(float(value)) > 0.0 for value in packet.physical_action[:7])
        return (
            motion
            or packet.gripper_command_changed
            or packet.episode_command.upper() != "CONTINUE"
        )

    def _pop_action_request(self) -> G2RemoteKeyboardPacket:
        """Deliver exactly one relative delta at one policy boundary.

        Relative IK commands are not safely coalescible under the downstream
        joint-target rate limiter.  For example, three queued +3 mm Z requests
        merged into one +9 mm request, but the limiter allowed only the same
        one-step motion as a single request and silently discarded the other
        two operator inputs when the neutral command followed.
        """

        return self._pending_action_packets.popleft()

    def clear_pending_actions(self) -> None:
        """Drop episode-local delta/edge requests at a runtime reset boundary."""

        self._pending_action_packets.clear()
        self._last_delivery_was_delta = False

    def sample(
        self,
        *,
        now_monotonic_s: float,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
        motion_queue_cleared = False
        while True:
            line = self._stream.readline()
            if not line:
                break
            try:
                packet = self._packet_from_payload(json.loads(line))
            except (G2RemoteCommandError, json.JSONDecodeError) as exc:
                raise G2RemoteCommandError("REMOTE_COMMAND_JSON_INVALID") from exc
            self._ingest_validator.submit(
                packet,
                received_monotonic_s=now_monotonic_s,
                device=device,
            )
            self._latest_packet = packet
            if packet.clear_motion_queue:
                self._pending_action_packets.clear()
                self._last_delivery_was_delta = False
                motion_queue_cleared = True
            elif self._requires_fifo_delivery(packet):
                self._pending_action_packets.append(packet)
                if len(self._pending_action_packets) > self._maximum_pending_action_packets:
                    raise G2RemoteCommandError("REMOTE_COMMAND_ACTION_QUEUE_OVERFLOW")

        selected: G2RemoteKeyboardPacket | None = None
        coalesced_packet_count = 0
        if self._pending_action_packets:
            selected = self._pop_action_request()
            coalesced_packet_count = 1
        elif (
            self._latest_packet is not None
            and self._latest_packet.sequence_id != self._last_delivered_sequence_id
        ):
            selected = self._latest_packet
        if selected is not None:
            self.adapter.submit(
                selected,
                received_monotonic_s=now_monotonic_s,
                device=device,
            )
            self._last_delivered_sequence_id = selected.sequence_id
            self._last_delivery_was_delta = any(
                abs(float(value)) > 0.0 for value in selected.physical_action[:7]
            )
        physical, normalized, metadata = self.adapter.sample(
            now_monotonic_s=now_monotonic_s
        )
        synthetic_neutral = False
        if selected is None and self._last_delivery_was_delta:
            # A physical delta is a one-policy-step request, not a velocity to
            # replay indefinitely while waiting for the next terminal line.
            physical[:7] = 0.0
            normalized = self.adapter.normalization.normalize(
                physical.unsqueeze(0)
            ).squeeze(0)
            self._last_delivery_was_delta = False
            synthetic_neutral = True
        metadata.update(
            {
                "transport_delivery": "ORDERED_FIFO_ONE_RELATIVE_PULSE_PER_POLICY_STEP_V3",
                "gripper_command_changed": int(selected.gripper_command_changed)
                if selected is not None
                else 0,
                "pending_action_packets": len(self._pending_action_packets),
                "coalesced_packet_count": coalesced_packet_count,
                "synthetic_neutral_after_pulse": int(synthetic_neutral),
                "motion_queue_cleared": int(motion_queue_cleared),
                # View selection is latest-state control, not a motion delta;
                # it must not wait behind the bounded motion FIFO.
                "operator_view": (
                    self._latest_packet.operator_view
                    if self._latest_packet is not None
                    else metadata.get("operator_view", "right_wrist")
                ),
            }
        )
        return physical, normalized, metadata

    def close(self) -> None:
        self._stream.close()


__all__ = [
    "G2_REMOTE_KEYBOARD_PACKET_SCHEMA",
    "G2_REMOTE_KEYBOARD_ACTION_ORDER",
    "G2_REMOTE_EPISODE_COMMANDS",
    "G2_REMOTE_OPERATOR_VIEWS",
    "G2_REMOTE_COMMAND_FRAMES",
    "G2RemoteCommandError",
    "G2RemoteCommandStale",
    "G2RemoteKeyboardCommandAdapter",
    "G2RemoteJsonlCommandSource",
    "G2RemoteKeyboardPacket",
]
