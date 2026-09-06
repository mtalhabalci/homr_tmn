import argparse
import os
import shutil
import sys
from typing import Any

import torch
import torch._dynamo
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from homr import download_utils
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.architecture.transformer.tromr_arch import TrOMR, load_model
from training.datasets.convert_grandstaff import (
    convert_grandstaff,
    grandstaff_train_index,
)
from training.datasets.convert_lieder import convert_lieder, lieder_train_index
from training.datasets.convert_primus import convert_primus_dataset, primus_train_index
from training.datasets.convert_symbtr import (
    convert_symbtr,
    symbtr_train_index,
    symbtr_val_index,
)
from training.run_id import get_run_id
from training.transformer.data_loader import label_names, load_dataset
from training.transformer.metrics import HomrTrainer
from training.transformer.mix_datasets import mix_training_sets

torch._dynamo.config.suppress_errors = True


class FreezeCallback(TrainerCallback):
    """
    Callback to freeze the backbone for a set number of epochs.
    Standard practice is ~2 epochs.
    """

    def __init__(self, epochs_to_freeze: int = 2):
        self.epochs_to_freeze = epochs_to_freeze
        self._backbone_frozen = False

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        model = kwargs.get("model")
        if model and hasattr(model, "freeze_backbone"):
            eprint(f"Freezing backbone for the first {self.epochs_to_freeze} epochs")
            model.freeze_backbone()
            self._backbone_frozen = True

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        model = kwargs.get("model")
        if model and self._backbone_frozen and state.epoch and state.epoch >= self.epochs_to_freeze:
            eprint(f"Unfreezing backbone at epoch {state.epoch}")
            model.unfreeze_backbone()
            self._backbone_frozen = False


def download_training_checkpoint(config: Config) -> None:
    """Fetch the PyTorch weights fine-tuning starts from.

    homr.main downloads the ONNX models used for inference; the .pth the trainer
    needs lives in a different release and nothing fetched it, which made a fresh
    machine (a Colab runtime, say) unable to fine-tune at all.
    """
    checkpoint = config.filepaths.checkpoint
    if os.path.exists(checkpoint):
        return
    base_url = "https://github.com/liebharc/homr/releases/download/checkpoints/"
    name = os.path.basename(checkpoint).removesuffix(".pth")
    destination = os.path.dirname(checkpoint)
    os.makedirs(destination, exist_ok=True)
    archive = os.path.join(destination, name + ".zip")
    eprint(f"Downloading training checkpoint {name} - this is only required once")
    try:
        download_utils.download_file(base_url + name + ".zip", archive)
        download_utils.unzip_file(archive, destination)
    finally:
        if os.path.exists(archive):
            os.remove(archive)
    if not os.path.exists(checkpoint):
        eprint(f"Checkpoint still missing after download: {checkpoint}")
        sys.exit(1)


def seed_makam_accidentals(model: TrOMR, config: Config) -> None:
    """Start the makam accidentals from the Western ones they replace.

    Grown-in tokens begin from random weights while the tokens they supersede
    carry years of training, so the model keeps answering "#" to a sharp it is
    now meant to call sharp4. After three epochs every accidental class still
    scored zero recall, with 483 sharps read as "#".

    The glyphs are the same shape: an ordinary sharp is the 4-comma sharp, an
    ordinary flat the 5-comma one. Copying the old rows into the new ones hands
    the model that knowledge instead of making it rediscover it, leaving only
    the comma distinctions to learn. The Western accidentals are then pushed out
    of reach, since makam labels never use them and a tie would otherwise be
    settled by whichever weights are better trained.
    """
    vocab = config.lift_vocab
    net = model.decoder.net
    with torch.no_grad():
        for token, index in vocab.items():
            source = "#" if token.startswith("sharp") else "b" if token.startswith("flat") else None
            if source is None or source not in vocab:
                continue
            origin = vocab[source]
            net.lift_emb.emb.weight[index] = net.lift_emb.emb.weight[origin]
            net.to_logits_lift.weight[index] = net.to_logits_lift.weight[origin]
            net.to_logits_lift.bias[index] = net.to_logits_lift.bias[origin]
        unused = [vocab[t] for t in ("#", "##", "b", "bb") if t in vocab]
        for index in unused:
            net.to_logits_lift.bias[index] = -1e4
    eprint(f"Seeded {len(vocab) - 7} makam accidentals and retired {len(unused)} Western ones")


def load_training_index(file_path: str) -> list[str]:
    with open(file_path) as f:
        return f.readlines()


def check_data_source(all_file_paths: list[str]) -> bool:
    result = True
    for file_paths in all_file_paths:
        paths = file_paths.strip().split(",")
        for path in paths:
            if path == "nosymbols":
                continue
            if not os.path.exists(path):
                eprint(f"Index {file_paths} does not exist due to {path}")
                result = False
    return result


def load_and_mix_training_sets(
    index_paths: list[str], weights: list[float], number_of_files: int
) -> list[str]:
    if len(index_paths) != len(weights):
        eprint("Error: Number of index paths and weights do not match")
        sys.exit(1)
    data_sources = [load_training_index(index) for index in index_paths]
    if not all(check_data_source(data) for data in data_sources):
        eprint("Error in datasets found")
        sys.exit(1)
    eprint(
        "Total number of training files to choose from", sum([len(data) for data in data_sources])
    )
    return mix_training_sets(data_sources, weights, number_of_files)


script_location = os.path.dirname(os.path.realpath(__file__))

git_root = os.path.join(script_location, "..", "..")


def _check_datasets_are_present(selected_datasets: list[str]) -> list[str]:
    for dataset in selected_datasets:
        if dataset == primus_train_index and not os.path.exists(primus_train_index):
            convert_primus_dataset()

        if dataset == grandstaff_train_index and not os.path.exists(grandstaff_train_index):
            convert_grandstaff()

        if dataset == lieder_train_index and not os.path.exists(lieder_train_index):
            convert_lieder()

        if dataset == symbtr_train_index and not os.path.exists(symbtr_train_index):
            convert_symbtr()
    return selected_datasets


def train_transformer(
    fp32: bool = False,
    resume: str = "",
    smoke_test: bool = False,
    fine_tune: bool = False,
    epochs: int | None = None,
    limit: int | None = None,
    lift_only: bool = False,
    work_dir: str | None = None,
    lr: float | None = None,
    full: bool = False,
) -> None:
    number_of_epochs = 35
    if smoke_test:
        number_of_epochs = 10
    elif fine_tune:
        number_of_epochs = 15
    if epochs is not None:
        number_of_epochs = epochs
    resume_from_checkpoint = None
    validation_index: list[str] | None = None

    # A Colab runtime can disappear mid-run, taking hours of training with it.
    # Point work_dir at mounted Drive and both the per-epoch checkpoints and the
    # finished model survive, so --resume can pick the run back up.
    checkpoint_folder = os.path.join(work_dir, "current_training") if work_dir \
        else "current_training"
    if resume:
        resume_from_checkpoint = (
            os.path.join(checkpoint_folder, resume) if work_dir
            else os.path.join(git_root, checkpoint_folder, resume)
        )
    elif not work_dir and os.path.exists(os.path.join(git_root, checkpoint_folder)):
        shutil.rmtree(os.path.join(git_root, checkpoint_folder))
    if work_dir:
        os.makedirs(checkpoint_folder, exist_ok=True)

    if smoke_test:
        number_of_files = -1
        train_index = load_and_mix_training_sets(
            _check_datasets_are_present(
                [lieder_train_index, grandstaff_train_index, primus_train_index]
            ),
            [1.0, 1.0, 1.0],
            number_of_files,
        )
    elif fine_tune:
        # Makam fine-tuning runs on SymbTr alone. The two repertoires disagree
        # about what the ordinary sharp glyph means - a semitone in Western
        # notation, four commas in makam - so mixing them would teach the lift
        # head two labels for one picture.
        number_of_files = -1
        train_index = load_and_mix_training_sets(
            _check_datasets_are_present([symbtr_train_index]), [1.0], number_of_files
        )
        # SymbTr is split by work up front, so use that split rather than
        # slicing the shuffled, oversampled training list.
        validation_index = load_training_index(symbtr_val_index)
        if limit is not None:
            # A short end-to-end check: a slice of both sides, not a real run.
            train_index = train_index[:limit]
            validation_index = validation_index[: max(1, limit // 10)]
    else:
        number_of_files = -1
        train_index = load_and_mix_training_sets(
            _check_datasets_are_present(
                [lieder_train_index, grandstaff_train_index, primus_train_index]
            ),
            [1.0, 1.0, 1.0],
            number_of_files,
        )

    config = Config()
    datasets = load_dataset(
        train_index, config, val_split=0.1, validation_samples=validation_index
    )

    compile_threshold = 50000
    compile_model = (
        number_of_files < 0 or number_of_files * number_of_epochs >= compile_threshold
    )  # Compiling needs time, but pays off for large datasets
    if limit is not None:
        # A limited run exists to prove the pipeline turns over. Compiling would
        # spend several minutes warming up and swamp the run it is meant to check.
        compile_model = False
    if compile_model:
        eprint("Compiling model")

    # Fine-tuning normally nudges weights that are already close. Here whole
    # output classes are new, so the default 1e-5 barely moves them.
    if lr is not None:
        learning_rate = lr
    elif not fine_tune:
        learning_rate = 1e-4
    elif full:
        # Every weight moves now, including features that took the whole
        # Western corpus to learn, so step an order of magnitude smaller than
        # when only the freshly seeded output layers were free.
        learning_rate = 2e-5
    else:
        learning_rate = 5e-5
    eprint(f"Learning rate {learning_rate}")

    run_id = get_run_id()

    batch_size = 6 if fp32 else 18

    train_args = TrainingArguments(
        checkpoint_folder,
        torch_compile=compile_model,
        eval_strategy="epoch",
        save_strategy="epoch",
        # Full checkpoints carry the optimiser state and run to a gigabyte;
        # keep only what a resume needs plus the best model.
        save_total_limit=2,
        learning_rate=learning_rate,
        optim="adamw_torch_fused",
        gradient_accumulation_steps=4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        num_train_epochs=number_of_epochs,
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=["tensorboard"],
        logging_dir=os.path.join("logs", f"run{run_id}"),
        label_names=label_names,
        bf16=not fp32,
        dataloader_pin_memory=True,
        # Colab runtimes vary from 2 to 12 vCPUs; asking for more workers than
        # there are cores starves the loaders instead of filling the GPU.
        dataloader_num_workers=min(12, max(2, (os.cpu_count() or 4) - 2)),
    )

    if fine_tune:
        eprint("Fine tuning model from", config.filepaths.checkpoint)
        download_training_checkpoint(config)
        model = load_model(config)
        seed_makam_accidentals(model, config)
        if full:
            # Freezing everything but the output layers leaves 0.5% of the
            # network trainable: a linear readout of features a Western model
            # learned. That is enough to relabel a sharp as sharp4, and never
            # enough to tell a one-comma sharp from a four-comma one or to read
            # the small digit that separates flat1 from flat2, because those
            # distinctions are not in the frozen features to begin with. They
            # stalled between zero and forty percent while everything inferable
            # from position went past ninety.
            for param in model.parameters():
                param.requires_grad = True
            eprint("Training the whole network")
        elif lift_only:
            model.freeze_encoder()
            model.freeze_decoder()
            model.unfreeze_lift_decoder()
            eprint("Training the lift branch only")
        else:
            # The makam key signature and the usul are rhythm tokens, so the
            # rhythm branch has to learn too. Pass --lift-only for the safer,
            # narrower run that cannot disturb how symbols are read at all.
            model.freeze_encoder()
            model.freeze_decoder()
            model.unfreeze_lift_decoder()
            model.unfreeze_rhythm_decoder()
            eprint("Training the lift and rhythm branches")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        eprint(f"Trainable parameters: {trainable:,} of {total:,}")
    else:
        model = TrOMR(config)

    model_name = "pytorch_model"

    model_destination = os.path.join(
        work_dir if work_dir
        else os.path.join(git_root, "training", "architecture", "transformer"),
        f"{model_name}_{run_id}.pth",
    )

    if os.path.exists(model_destination):
        eprint("Model already exists", model_destination)
        return

    try:
        callbacks: list[TrainerCallback] = [EarlyStoppingCallback(early_stopping_patience=5)]
        if not fine_tune:
            callbacks.append(FreezeCallback(epochs_to_freeze=2))

        trainer = HomrTrainer(
            model,
            train_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            callbacks=callbacks,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        eprint("Interrupted")
    torch.save(model.state_dict(), model_destination)
    eprint(f"Saved model to {model_destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the transformer")
    parser.add_argument(
        "--fine", action="store_true", help="Fine-tune the makam accidentals on SymbTr."
    )
    parser.add_argument("--fp32", action="store_true", help="Train in fp32 instead of bf16.")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint folder to resume from.")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the number of epochs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N staff samples. For a quick end-to-end check.",
    )
    parser.add_argument(
        "--lift-only",
        action="store_true",
        help="Train only the accidental branch, leaving rhythm reading untouched.",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override the learning rate."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train every weight, not just the output layers. Needed for glyph "
             "distinctions the Western encoder never had to make.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Where to keep checkpoints and the finished model. Point it at "
             "mounted Drive so a lost Colab runtime does not lose the run.",
    )
    options = parser.parse_args()
    if options.fine:
        train_transformer(
            fp32=options.fp32,
            resume=options.resume,
            fine_tune=True,
            epochs=options.epochs,
            limit=options.limit,
            lift_only=options.lift_only,
            work_dir=options.work_dir,
            lr=options.lr,
            full=options.full,
        )
    else:
        train_transformer(smoke_test=True)
