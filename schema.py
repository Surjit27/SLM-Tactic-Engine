"""Schema reference for synthetic Farcana-style dataset tables."""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class FieldSpec:
    name: str
    start: int
    end: int
    shape: str
    value_range: str
    description: str


@dataclass(frozen=True)
class ModelInterfaceSpec:
    model_name: str
    trainable_parameters: str
    input_fields: str
    predicted_fields: str
    output_shape: str


OBSERVATION_SCHEMA: list[FieldSpec] = [
    FieldSpec("player_state", 0, 15, "(16,)", "mixed [-1,1] and [0,1]", "Player position, velocity, health, armor, ammo, phase, aim."),
    FieldSpec("enemy_data", 16, 65, "(50,)", "mostly [-1,1]/[0,1]", "Up to 10 enemies, 5 features each (relative position, distance, angle)."),
    FieldSpec("cover_data", 66, 75, "(10,)", "[-1,1]/[0,1]", "Up to 2 nearest cover points, 5 features each."),
    FieldSpec("teammate_data", 76, 85, "(10,)", "[-1,1]/[0,1]", "Up to 2 nearest teammates, 5 features each."),
    FieldSpec("audio_threat", 86, 95, "(10,)", "[0,1] and [-1,1]", "Threat/audio/zone/utilities indicators."),
    FieldSpec("temporal_history", 96, 255, "(160,)", "typically [-1,1]", "Rolling buffer of prior action vectors (5 x 32)."),
]


ACTION_SCHEMA: list[FieldSpec] = [
    FieldSpec("movement", 0, 4, "(5,)", "mixed [-1,1] and [0,1]", "Forward, strafe, stance, jump, sprint."),
    FieldSpec("aim", 5, 8, "(4,)", "mixed [-1,1] and [0,1]", "Pitch delta, yaw delta, ADS, aim confidence."),
    FieldSpec("combat_core", 9, 14, "(6,)", "[0,1]", "Fire, fire mode, reload, melee, grenade, utility trigger."),
    FieldSpec("misc_actions", 15, 19, "(5,)", "[-1,1] or [0,1]", "Reserved slots for additional controls."),
    FieldSpec("tactical_one_hot", 20, 25, "(6,)", "{0,1}", "One-hot tactical decision used by RL example."),
    FieldSpec("fusion_tail", 26, 31, "(6,)", "[-1,1]", "Remaining fused controls (placeholder in synthetic data)."),
]


TACTICAL_LABELS: dict[int, str] = {
    0: "push",
    1: "hold",
    2: "flank",
    3: "retreat",
    4: "heal",
    5: "ability",
}


DATA_TABLE_SCHEMA = {
    "observations": {
        "dtype": "float32",
        "shape": f"(N, {config.OBS_SIZE})",
        "description": "Model input tensor; each row is one encoded game state.",
    },
    "actions": {
        "dtype": "float32",
        "shape": f"(N, {config.ACTION_SIZE})",
        "description": "Synthetic fused action tensor aligned to architecture design.",
    },
    "tactical_targets": {
        "dtype": "int64",
        "shape": "(N,)",
        "description": "Discrete tactical class target in range [0..5].",
    },
}

MODEL_INTERFACE_SCHEMA: list[ModelInterfaceSpec] = [
    ModelInterfaceSpec(
        model_name="slm_tactical_planner",
        trainable_parameters="Numeric projection weights, sequence/attention blocks, MLP blocks, tactical classifier head (no token embedding table)",
        input_fields="obs[0:256] + temporal summary from obs[96:256]",
        predicted_fields="tactical logits/probabilities for classes [push, hold, flank, retreat, heal, ability]",
        output_shape="(6,)",
    ),
    ModelInterfaceSpec(
        model_name="aim_model",
        trainable_parameters="Sequence/MLP weights for aim deltas and confidence head",
        input_fields="obs[0:256] + tactical embedding (from SLM)",
        predicted_fields="aim_delta_pitch, aim_delta_yaw, ads, aim_target_lock (action[5:9])",
        output_shape="(4,)",
    ),
    ModelInterfaceSpec(
        model_name="movement_model",
        trainable_parameters="Transformer/MLP weights for locomotion control head",
        input_fields="obs[0:256] + tactical embedding (from SLM)",
        predicted_fields="move_forward, move_strafe, stance, jump, sprint (action[0:5])",
        output_shape="(5,)",
    ),
    ModelInterfaceSpec(
        model_name="weapon_model",
        trainable_parameters="MLP weights for weapon preference classification",
        input_fields="obs[0:256]",
        predicted_fields="weapon slot logits/probabilities -> mapped into action control fields",
        output_shape="(6,)",
    ),
    ModelInterfaceSpec(
        model_name="fusion_model",
        trainable_parameters="Fusion MLP weights combining all specialist outputs",
        input_fields="concat(tactical, aim, movement, weapon outputs)",
        predicted_fields="final fused action tensor",
        output_shape="(32,)",
    ),
]


def print_schema() -> None:
    """Pretty-print table-like schema details for quick understanding."""
    print("=== Dataset Table Schema ===")
    for key, spec in DATA_TABLE_SCHEMA.items():
        print(f"- {key}: dtype={spec['dtype']}, shape={spec['shape']}")
        print(f"  {spec['description']}")

    print("\n=== Observation[256] Segments ===")
    for field in OBSERVATION_SCHEMA:
        print(f"- {field.name:16s} [{field.start:3d}..{field.end:3d}] {field.shape:7s} {field.value_range}")
        print(f"  {field.description}")

    print("\n=== Action[32] Segments ===")
    for field in ACTION_SCHEMA:
        print(f"- {field.name:16s} [{field.start:3d}..{field.end:3d}] {field.shape:7s} {field.value_range}")
        print(f"  {field.description}")

    print("\n=== Tactical Labels ===")
    for idx, name in TACTICAL_LABELS.items():
        print(f"- {idx}: {name}")

    print("\n=== Model Interfaces (Input -> Parameters -> Prediction) ===")
    for spec in MODEL_INTERFACE_SCHEMA:
        print(f"- {spec.model_name}")
        print(f"  trainable_parameters: {spec.trainable_parameters}")
        print(f"  input_fields:         {spec.input_fields}")
        print(f"  predicted_fields:     {spec.predicted_fields}")
        print(f"  output_shape:         {spec.output_shape}")


if __name__ == "__main__":
    print_schema()
