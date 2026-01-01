from helpers import training


class Dummy:
    def __init__(self, value=None):
        self.value = value


def _make_controls_for_hf():
    def D(v=""):
        return Dummy(v)

    return {
        "train_source": D("Hugging Face"),
        "train_hf_repo": D("username/ds"),
        "train_hf_split": D("train"),
        "train_hf_config": D(""),
        "train_json_path": D(""),
        "train_db_session_dd": D(""),
        "base_model": D("base-model"),
        "out_dir_tf": D("/data/out"),
        "epochs_tf": D("5"),
        "lr_tf": D("1e-4"),
        "batch_tf": D("4"),
        "grad_acc_tf": D("2"),
        "max_steps_tf": D("100"),
        "use_lora_cb": Dummy(True),
        "lora_r_dd": D("16"),
        "lora_alpha_tf": D("32"),
        "lora_dropout_tf": D("0"),
        "use_rslora_cb": Dummy(False),
        "packing_cb": Dummy(True),
        "auto_resume_cb": Dummy(False),
        "push_cb": Dummy(True),
        "hf_repo_id_tf": D("username/model"),
        "resume_from_tf": D("checkpoint"),
        "warmup_steps_tf": D("10"),
        "weight_decay_tf": D("0.01"),
        "lr_sched_dd": D("cosine"),
        "optim_dd": D("adamw"),
        "logging_steps_tf": D("5"),
        "logging_first_step_cb": Dummy(True),
        "disable_tqdm_cb": Dummy(False),
        "seed_tf": D("42"),
        "save_strategy_dd": D("steps"),
        "save_total_limit_tf": D("3"),
        "pin_memory_cb": Dummy(True),
        "report_to_dd": D("wandb"),
        "fp16_cb": Dummy(False),
        "bf16_cb": Dummy(True),
    }


def test_build_hp_from_controls_hf_dataset():
    controls = _make_controls_for_hf()
    hp = training.build_hp_from_controls(**controls)

    assert hp["base_model"] == "base-model"
    assert hp["epochs"] == "5"
    assert hp["lr"] == "1e-4"
    assert hp["bsz"] == "4"
    assert hp["grad_accum"] == "2"
    assert hp["max_steps"] == "100"

    assert hp["hf_dataset_id"] == "username/ds"
    assert hp["hf_dataset_split"] == "train"

    assert hp["use_lora"] is True
    assert hp["packing"] is True
    assert hp["push"] is True
    assert hp["hf_repo_id"] == "username/model"
    assert hp["resume_from"] == "checkpoint"
    assert hp["optim"] == "adamw"


def test_build_hp_from_controls_database_source():
    controls = _make_controls_for_hf()
    controls["train_source"].value = "Database"
    controls["train_hf_repo"].value = ""
    controls["train_db_session_dd"].value = "123"

    hp = training.build_hp_from_controls(**controls)

    assert "hf_dataset_id" not in hp
    assert hp["db_session_id"] == "123"


def test_build_hp_default_values():
    """Test default values when controls are empty."""

    def D(v=""):
        return Dummy(v)

    controls = {
        "train_source": D("Hugging Face"),
        "train_hf_repo": D(""),
        "train_hf_split": D(""),
        "train_hf_config": D(""),
        "train_json_path": D(""),
        "train_db_session_dd": D(""),
        "base_model": D(""),
        "out_dir_tf": D(""),
        "epochs_tf": D(""),
        "lr_tf": D(""),
        "batch_tf": D(""),
        "grad_acc_tf": D(""),
        "max_steps_tf": D(""),
        "use_lora_cb": Dummy(False),
        "lora_r_dd": D(""),
        "lora_alpha_tf": D(""),
        "lora_dropout_tf": D(""),
        "use_rslora_cb": Dummy(False),
        "packing_cb": Dummy(False),
        "auto_resume_cb": Dummy(False),
        "push_cb": Dummy(False),
        "hf_repo_id_tf": D(""),
        "resume_from_tf": D(""),
        "warmup_steps_tf": D(""),
        "weight_decay_tf": D(""),
        "lr_sched_dd": D(""),
        "optim_dd": D(""),
        "logging_steps_tf": D(""),
        "logging_first_step_cb": Dummy(False),
        "disable_tqdm_cb": Dummy(False),
        "seed_tf": D(""),
        "save_strategy_dd": D(""),
        "save_total_limit_tf": D(""),
        "pin_memory_cb": Dummy(False),
        "report_to_dd": D(""),
        "fp16_cb": Dummy(False),
        "bf16_cb": Dummy(False),
    }

    hp = training.build_hp_from_controls(**controls)

    # Should have default values
    assert hp["epochs"] == "3"
    assert hp["lr"] == "2e-4"
    assert hp["bsz"] == "2"
    assert hp["grad_accum"] == "4"
    assert hp["max_steps"] == "200"
    assert "unsloth" in hp["base_model"]


def test_build_hp_filters_unallowed_keys():
    """Test that unallowed keys are filtered out."""
    controls = _make_controls_for_hf()
    hp = training.build_hp_from_controls(**controls)

    # These keys ARE now allowed and should be present when set
    assert "warmup_steps" in hp
    assert hp["warmup_steps"] == "10"
    assert "weight_decay" in hp
    assert hp["weight_decay"] == "0.01"
    assert "lr_scheduler" in hp
    assert hp["lr_scheduler"] == "cosine"

    # These keys should still be filtered out (not in _allowed set)
    assert "logging_steps" not in hp
    assert "seed" not in hp
    assert "save_strategy" not in hp
    assert "pin_memory" not in hp
    # fp16/bf16 only added when checkbox is True; bf16 is True in test data
    assert "fp16" not in hp  # fp16_cb is False
    assert "bf16" not in hp  # bf16 is not in _allowed set


def test_build_hp_optimizer_renamed():
    """Test that optimizer is renamed to optim."""
    controls = _make_controls_for_hf()
    hp = training.build_hp_from_controls(**controls)

    assert "optimizer" not in hp
    assert "optim" in hp
    assert hp["optim"] == "adamw"


def test_build_hp_with_hf_config():
    """Test HF dataset config is set but filtered by allowed keys."""
    controls = _make_controls_for_hf()
    controls["train_hf_config"].value = "main"

    hp = training.build_hp_from_controls(**controls)

    # hf_dataset_config is not in the allowed keys list, so it's filtered out
    # This is expected behavior - the function filters to only allowed keys
    assert "hf_dataset_config" not in hp


def test_build_hp_lora_disabled():
    """Test with LoRA disabled."""
    controls = _make_controls_for_hf()
    controls["use_lora_cb"].value = False

    hp = training.build_hp_from_controls(**controls)

    assert hp["use_lora"] is False


def test_build_hp_packing_disabled():
    """Test with packing disabled."""
    controls = _make_controls_for_hf()
    controls["packing_cb"].value = False

    hp = training.build_hp_from_controls(**controls)

    assert "packing" not in hp


def test_build_hp_auto_resume_enabled():
    """Test with auto resume enabled."""
    controls = _make_controls_for_hf()
    controls["auto_resume_cb"].value = True

    hp = training.build_hp_from_controls(**controls)

    assert hp.get("auto_resume") is True


def test_build_hp_push_disabled():
    """Test with push disabled."""
    controls = _make_controls_for_hf()
    controls["push_cb"].value = False

    hp = training.build_hp_from_controls(**controls)

    assert "push" not in hp


def test_build_hp_no_resume_from():
    """Test without resume checkpoint."""
    controls = _make_controls_for_hf()
    controls["resume_from_tf"].value = ""

    hp = training.build_hp_from_controls(**controls)

    assert "resume_from" not in hp


def test_build_hp_no_hf_repo_id():
    """Test without HF repo ID."""
    controls = _make_controls_for_hf()
    controls["hf_repo_id_tf"].value = ""

    hp = training.build_hp_from_controls(**controls)

    assert "hf_repo_id" not in hp


def test_build_hp_whitespace_handling():
    """Test that whitespace is stripped."""
    controls = _make_controls_for_hf()
    controls["base_model"].value = "  model-name  "
    controls["train_hf_repo"].value = "  user/repo  "

    hp = training.build_hp_from_controls(**controls)

    assert hp["base_model"] == "model-name"
    assert hp["hf_dataset_id"] == "user/repo"
