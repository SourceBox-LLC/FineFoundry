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
        "base_model": D("base-model"),
        "out_dir_tf": D("/data/out"),
        "epochs_tf": D("5"),
        "lr_tf": D("1e-4"),
        "batch_tf": D("4"),
        "grad_acc_tf": D("2"),
        "max_steps_tf": D("100"),
        "use_lora_cb": Dummy(True),
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


def test_build_hp_from_controls_json_path_source():
    controls = _make_controls_for_hf()
    controls["train_source"].value = "JSON file"
    controls["train_hf_repo"].value = ""
    controls["train_json_path"].value = "/data/train.json"

    hp = training.build_hp_from_controls(**controls)

    assert "hf_dataset_id" not in hp
    assert hp["json_path"] == "/data/train.json"
