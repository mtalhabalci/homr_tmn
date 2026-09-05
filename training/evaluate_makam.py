"""Evaluate a fine-tuned model on the SymbTr test split, class by class.

The single accuracy figure the trainer reports is not usable on its own here:
three quarters of the accidental labels are "nothing drawn", so a model that
never predicts an accidental at all already scores about 75%. What matters is
how each symbol fares individually, and what it gets confused with.

    python -m training.evaluate_makam
    python -m training.evaluate_makam --checkpoint path/to/model.pth --limit 300

Reports, for every branch, per-class recall and precision, and for the
accidentals also the confusions that actually happen. Symbols the model drops
or invents are counted separately, since not losing symbols is the priority.
"""

import argparse
import collections
import os
import sys

import torch

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.architecture.transformer.tromr_arch import TrOMR, load_model
from training.datasets.convert_symbtr import symbtr_test_index
from training.transformer.data_loader import load_dataset

BRANCHES = ("rhythm", "pitch", "lift", "position", "articulations")


def _vocabularies(config: Config) -> dict[str, dict[str, int]]:
    return {
        "rhythm": config.rhythm_vocab,
        "pitch": config.pitch_vocab,
        "lift": config.lift_vocab,
        "position": config.position_vocab,
        "articulations": config.articulation_vocab,
    }


def evaluate(checkpoint: str | None, limit: int | None, batch_size: int) -> None:
    config = Config()
    if checkpoint:
        config.filepaths.checkpoint = checkpoint
    if not os.path.exists(config.filepaths.checkpoint):
        eprint(f"No checkpoint at {config.filepaths.checkpoint}")
        sys.exit(1)
    if not os.path.exists(symbtr_test_index):
        eprint(f"No test index at {symbtr_test_index}. Run convert_symbtr first.")
        sys.exit(1)

    with open(symbtr_test_index, encoding="utf-8") as handle:
        samples = [line for line in handle if line.strip()]
    if limit:
        samples = samples[:limit]
    eprint(f"Evaluating {len(samples)} staff samples from {symbtr_test_index}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config)
    model.eval_mode()

    data = load_dataset(samples, config, validation_samples=samples)
    loader = torch.utils.data.DataLoader(
        data["validation"], batch_size=batch_size, shuffle=False
    )

    hits: dict[str, collections.Counter] = {b: collections.Counter() for b in BRANCHES}
    truth: dict[str, collections.Counter] = {b: collections.Counter() for b in BRANCHES}
    guessed: dict[str, collections.Counter] = {b: collections.Counter() for b in BRANCHES}
    confusion: collections.Counter = collections.Counter()

    names = {
        branch: {index: token for token, index in vocab.items()}
        for branch, vocab in _vocabularies(config).items()
    }

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            outputs = model(
                batch["inputs"],
                batch["rhythms"],
                batch["pitchs"],
                batch["lifts"],
                batch["articulations"],
                batch["positions"],
                batch["mask"],
            )
            logits = outputs["logits"]
            eval_mask = batch["mask"][:, 1:]
            for branch, branch_logits in zip(BRANCHES, logits):
                labels = batch[
                    {"rhythm": "rhythms", "pitch": "pitchs", "lift": "lifts",
                     "position": "positions", "articulations": "articulations"}[branch]
                ][:, 1:]
                preds = branch_logits.argmax(dim=-1)
                length = min(preds.shape[1], labels.shape[1], eval_mask.shape[1])
                preds, labels = preds[:, :length], labels[:, :length]
                mask = (labels != -100) & eval_mask[:, :length]
                for predicted, actual in zip(preds[mask].tolist(), labels[mask].tolist()):
                    truth[branch][actual] += 1
                    guessed[branch][predicted] += 1
                    if predicted == actual:
                        hits[branch][actual] += 1
                    elif branch == "lift":
                        confusion[(actual, predicted)] += 1

    for branch in BRANCHES:
        total = sum(truth[branch].values())
        correct = sum(hits[branch].values())
        if not total:
            continue
        eprint(f"\n=== {branch}  {correct}/{total} = {100 * correct / total:.1f}% ===")
        eprint(f"{'symbol':<22}{'in test':>9}{'found':>8}{'recall':>9}{'precision':>11}")
        for index, seen in truth[branch].most_common(20):
            name = names[branch].get(index, str(index))
            got = hits[branch][index]
            said = guessed[branch][index]
            recall = 100 * got / seen
            precision = 100 * got / said if said else 0.0
            eprint(f"{name:<22}{seen:>9}{got:>8}{recall:>8.1f}%{precision:>10.1f}%")

    if confusion:
        eprint("\n=== accidentals: what is mistaken for what ===")
        lift_names = names["lift"]
        for (actual, predicted), count in confusion.most_common(15):
            eprint(
                f"   {lift_names.get(actual, actual):<12} read as "
                f"{lift_names.get(predicted, predicted):<12} {count:>6}"
            )

    # Symbols dropped or invented, the measure that matters most here.
    empty_index = config.lift_vocab.get("_")
    dropped = sum(
        count for (actual, predicted), count in confusion.items()
        if predicted == empty_index and actual != empty_index
    )
    invented = sum(
        count for (actual, predicted), count in confusion.items()
        if actual == empty_index and predicted != empty_index
    )
    real = sum(
        seen for index, seen in truth["lift"].items()
        if lift_is_symbol(names["lift"].get(index, ""))
    )
    eprint("\n=== accidentals as a group ===")
    eprint(f"   printed in the test set : {real}")
    eprint(f"   missed (read as blank)  : {dropped}")
    eprint(f"   invented (blank read as): {invented}")
    if real:
        eprint(f"   caught                  : {100 * (real - dropped) / real:.1f}%")


def lift_is_symbol(name: str) -> bool:
    return name.startswith(("sharp", "flat")) or name == "N"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Model to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Use only N staffs.")
    parser.add_argument("--batch-size", type=int, default=8)
    options = parser.parse_args()
    evaluate(options.checkpoint, options.limit, options.batch_size)
