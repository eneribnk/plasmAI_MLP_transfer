import torch
import torch.nn as nn
from .models import Model, CombinedModel


STRATEGIES = {
    "S0": {
        "name": "Scratch",
        "use_pretrained": False,
        "freeze_layers": [],
        "gradual_unfreeze": False,
        "lr_multiplier": 1.0,
        "description": "Random initialization, trained solely on target budget."
    },
    "S1": {
        "name": "Full Fine-Tuning",
        "use_pretrained": True,
        "freeze_layers": [],
        "gradual_unfreeze": False,
        "lr_multiplier": 0.1,
        "description": "Initialize compatible layers from source, train all layers with reduced learning rate."
    },
    "S2": {
        "name": "Feature Extraction",
        "use_pretrained": True,
        "freeze_layers": [0, 1, 2],
        "gradual_unfreeze": False,
        "lr_multiplier": 1.0,
        "description": "Freeze all hidden layers; train only the output head."
    },
    "S3": {
        "name": "Gradual Unfreezing",
        "use_pretrained": True,
        "freeze_layers": [0, 1, 2],
        "gradual_unfreeze": True,
        "lr_multiplier": 1.0,
        "description": "Begin with frozen hidden layers, unfreeze after unfreeze_epoch and halve lr."
    },
    "S4": {
        "name": "Geometry Layer",
        "use_pretrained": True,
        "freeze_layers": [],
        "gradual_unfreeze": False,
        "lr_multiplier": 0.1,
        "description": "Combined multi-head architecture utilizing pretrained point-specific models."
    }
}


def build_model_for_strategy(strategy_id, input_size, output_size, h1=20, num_layers=3,
                             pretrained_source_path=None, source_input_cols=None,
                             target_input_cols=None, output_activation=None,
                             device="cpu", pretrained_models=None):
    """
    Construct model and transfer weights according to chosen strategy and chemistry alignment.
    """
    strat = STRATEGIES.get(strategy_id)
    if strat is None:
        raise ValueError(f"Unknown strategy '{strategy_id}'. Expected one of {list(STRATEGIES.keys())}")

    if strategy_id == "S4":
        if pretrained_models is None:
            raise ValueError("S4 (Geometry layer) requires pretrained_models list.")
        model = CombinedModel(h1=h1, num_layers=num_layers, pretrained_models=pretrained_models)
        return model, strat

    # S0 .. S3 use the standard MLP
    initial_freeze = strat["freeze_layers"] if strat["use_pretrained"] else []
    model = Model(input_size=input_size, output_size=output_size, h1=h1,
                  num_layers=num_layers, freeze_layers=initial_freeze,
                  output_activation=output_activation)

    if strat["use_pretrained"] and pretrained_source_path:
        state_dict = torch.load(pretrained_source_path, map_location=device, weights_only=True)
        load_source_weights(model, state_dict, source_input_cols, target_input_cols)
        # Re-apply freezing after loading state dict
        if initial_freeze:
            model.set_frozen_layers(initial_freeze)

    return model, strat


def load_source_weights(target_model, source_state_dict, source_input_cols, target_input_cols):
    """
    Intelligently transfer weights from source to target model:
    - Input adapter (Method A): maps shared input columns, initializes new inputs to zero.
    - Hidden layers: transferred directly.
    - Output layer: transferred if dimensions and physics align, otherwise left newly initialized.
    """
    target_state = target_model.state_dict()
    
    # 1. Input Layer Transfer
    src_w0 = source_state_dict.get("layers.0.weight")
    src_b0 = source_state_dict.get("layers.0.bias")

    if src_w0 is not None:
        if source_input_cols == target_input_cols or src_w0.shape == target_state["layers.0.weight"].shape:
            target_state["layers.0.weight"] = src_w0.clone()
            target_state["layers.0.bias"] = src_b0.clone()
        else:
            # Map shared inputs
            h, target_in = target_state["layers.0.weight"].shape
            new_w0 = torch.zeros((h, target_in), dtype=torch.float32)
            shared_indices = []
            if source_input_cols and target_input_cols:
                for src_idx, col in enumerate(source_input_cols):
                    if col in target_input_cols:
                        tgt_idx = target_input_cols.index(col)
                        new_w0[:, tgt_idx] = src_w0[:, src_idx]
            target_state["layers.0.weight"] = new_w0
            target_state["layers.0.bias"] = src_b0.clone()

    # 2. Hidden Layers Transfer
    for k in list(source_state_dict.keys()):
        if k.startswith("layers.") and not k.startswith("layers.0."):
            if k in target_state and target_state[k].shape == source_state_dict[k].shape:
                target_state[k] = source_state_dict[k].clone()

    # 3. Output Layer Transfer
    src_out_w = source_state_dict.get("out.weight")
    src_out_b = source_state_dict.get("out.bias")
    if src_out_w is not None and "out.weight" in target_state:
        if src_out_w.shape == target_state["out.weight"].shape:
            target_state["out.weight"] = src_out_w.clone()
            target_state["out.bias"] = src_out_b.clone()
        else:
            # Dimension mismatch: keep random initialization for the new output head
            pass

    target_model.load_state_dict(target_state)
