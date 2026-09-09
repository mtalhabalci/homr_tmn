"""Evaluate a fine-tuned model on the SymbTr test split, class by class.

The single accuracy figure the trainer reports is not usable on its own here:
three quarters of the accidental labels are "nothing drawn", so a model that
never predicts an accidental at all already scores about 75%. What matters is
how each symbol fares individually, and what it gets confused with.

    python -m training.evaluate_makam
    python -m training.evaluate_makam --checkpoint path/to/model.pth --limit 300
    python -m training.evaluate_makam --checkpoint <the western .pth> --lenient
    python -m training.evaluate_makam --checkpoint path/to/model.pth --generate
    python -m training.evaluate_makam --index datasets/SymbTr-shifted/index_shift-2.txt

Reports, for every branch, per-class recall and precision, and for the
accidentals also the confusions that actually happen. Symbols the model drops
or invents are counted separately, since not losing symbols is the priority.

--lenient scores the checkpoint homr ships, which has no makam tokens at all.
Marking its every accidental wrong would overstate the gap, since a Western
sharp is the four-comma sharp and a Western flat the five-comma flat -- the same
glyph under another name. Under --lenient those two count as correct, so the
baseline is read as generously as it honestly can be.

--generate is the honest reading test. By default the model is scored with
teacher forcing: at every step it is handed the correct previous symbols and
only has to name the next one, so a mistake never costs it anything downstream.
Under --generate it reads the staff on its own from the first symbol to the
last, exactly as it would in use, and the two sequences are lined up by edit
distance -- which also exposes symbols invented or dropped outright, something
teacher forcing cannot even represent.

Per-branch figures flatter the model, though: they ask whether the duration was
right and, separately, whether the pitch was right. A player hears neither in
isolation -- a note is right only when its duration, its pitch and its
accidental are all right at once. The last section reports that joint figure,
per note and per staff.
"""

import argparse
import collections
import os
import sys

import torch

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import has_rhythm_symbol_a_position
from training.architecture.transformer.tromr_arch import TrOMR, load_model
from training.datasets.convert_symbtr import symbtr_test_index
from training.transformer.data_loader import load_dataset

BRANCHES = ("rhythm", "pitch", "lift", "position", "articulations")
# EncodedSymbol names one branch differently. Deriving the attribute from
# BRANCHES keeps a generated symbol and a labelled one in the same column order;
# writing the two tuples out by hand once put position where articulation was
# expected, which marked nearly every symbol wrong while the per-branch figures
# stayed healthy.
SYMBOL_ATTRIBUTE = {
    "rhythm": "rhythm",
    "pitch": "pitch",
    "lift": "lift",
    "position": "position",
    "articulations": "articulation",
}


def _vocabularies(config: Config) -> dict[str, dict[str, int]]:
    return {
        "rhythm": config.rhythm_vocab,
        "pitch": config.pitch_vocab,
        "lift": config.lift_vocab,
        "position": config.position_vocab,
        "articulations": config.articulation_vocab,
    }


def evaluate(
    checkpoint: str | None,
    limit: int | None,
    batch_size: int,
    lenient: bool = False,
    index: str | None = None,
) -> None:
    config = Config()
    if checkpoint:
        config.filepaths.checkpoint = checkpoint
    if not os.path.exists(config.filepaths.checkpoint):
        eprint(f"No checkpoint at {config.filepaths.checkpoint}")
        sys.exit(1)
    chosen = index or symbtr_test_index
    if not os.path.exists(chosen):
        eprint(f"No test index at {chosen}. Run convert_symbtr first.")
        sys.exit(1)

    with open(chosen, encoding="utf-8") as handle:
        samples = [line for line in handle if line.strip()]
    if limit:
        samples = samples[:limit]
    eprint(f"Evaluating {len(samples)} staff samples from {chosen}")

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
    # An accidental is the same shape wherever it sits, but the model only ever
    # meets some of them on one degree of the staff: 752 of the 758 five-comma
    # sharps in training are on F5. Splitting recall by the note underneath says
    # whether a weak class is genuinely unread or only unread where it is scarce.
    by_pitch: collections.Counter = collections.Counter()
    by_pitch_hit: collections.Counter = collections.Counter()
    joint: collections.Counter = collections.Counter()
    staff_errors: list[int] = []
    staff_misreads: list[int] = []

    names = {
        branch: {index: token for token, index in vocab.items()}
        for branch, vocab in _vocabularies(config).items()
    }
    # Rests, barlines and clefs are symbols too, but only these carry a pitch a
    # listener would notice going wrong.
    note_indices = [
        index for token, index in config.rhythm_vocab.items() if token.startswith("note")
    ]
    # A barline or a repeat sign fills its remaining four columns with the
    # not-applicable marker. Reproducing that marker is a matter of form, not of
    # reading the page, and nothing downstream looks at those columns -- so for
    # these rows only the symbol itself is judged.
    carries_detail = [
        index
        for token, index in config.rhythm_vocab.items()
        if has_rhythm_symbol_a_position(token)
    ]

    lift_remap = None
    if lenient:
        # The Western sharp is the four-comma sharp and the Western flat the
        # five-comma flat: same glyph, different name. Let a checkpoint that
        # only knows the Western names score them.
        pairs = {"#": "sharp4", "b": "flat5"}
        lift_remap = list(range(len(config.lift_vocab)))
        for western, makam in pairs.items():
            if western in config.lift_vocab and makam in config.lift_vocab:
                lift_remap[config.lift_vocab[western]] = config.lift_vocab[makam]
        lift_remap = torch.tensor(lift_remap, device=device)
        eprint(f"Lenient scoring: {', '.join(f'{k} counts as {v}' for k, v in pairs.items())}")

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
            step: dict[str, tuple] = {}
            for branch, branch_logits in zip(BRANCHES, logits):
                labels = batch[
                    {"rhythm": "rhythms", "pitch": "pitchs", "lift": "lifts",
                     "position": "positions", "articulations": "articulations"}[branch]
                ][:, 1:]
                preds = branch_logits.argmax(dim=-1)
                length = min(preds.shape[1], labels.shape[1], eval_mask.shape[1])
                preds, labels = preds[:, :length], labels[:, :length]
                if branch == "lift" and lift_remap is not None:
                    preds = lift_remap[preds]
                mask = (labels != -100) & eval_mask[:, :length]
                step[branch] = (preds, labels)
                for predicted, actual in zip(preds[mask].tolist(), labels[mask].tolist()):
                    truth[branch][actual] += 1
                    guessed[branch][predicted] += 1
                    if predicted == actual:
                        hits[branch][actual] += 1
                    elif branch == "lift":
                        confusion[(actual, predicted)] += 1

            lift_preds, lift_labels_all = step["lift"]
            pitch_preds_all, pitch_labels_all = step["pitch"]
            span = min(lift_preds.shape[1], pitch_labels_all.shape[1], eval_mask.shape[1])
            valid = eval_mask[:, :span] & (lift_labels_all[:, :span] != -100)
            for row, column in valid.nonzero(as_tuple=False).tolist():
                actual = int(lift_labels_all[row][column])
                name = names["lift"].get(actual, "")
                if not lift_is_symbol(name):
                    continue
                pitch = names["pitch"].get(int(pitch_labels_all[row][column]), "?")
                by_pitch[(name, pitch)] += 1
                if int(lift_preds[row][column]) == actual:
                    by_pitch_hit[(name, pitch)] += 1

            # A symbol is only usable if every branch got it right at once.
            width = min(step[branch][0].shape[1] for branch in BRANCHES)
            rhythm_labels = step["rhythm"][1][:, :width]
            here = eval_mask[:, :width] & (rhythm_labels != -100)
            agree = torch.ones_like(here)
            for branch in BRANCHES:
                preds, labels = step[branch]
                agree &= preds[:, :width] == labels[:, :width]
            sounds = torch.zeros_like(here)
            for index in note_indices:
                sounds |= rhythm_labels == index
            detailed = torch.zeros_like(here)
            for index in carries_detail:
                detailed |= rhythm_labels == index
            rhythm_preds = step["rhythm"][0][:, :width]
            usable = torch.where(detailed, agree, rhythm_preds == rhythm_labels)
            pitch_preds, pitch_labels = step["pitch"]
            lift_preds, lift_labels = step["lift"]
            audible = (pitch_preds[:, :width] == pitch_labels[:, :width]) & (
                lift_preds[:, :width] == lift_labels[:, :width]
            )

            joint["symbols"] += int(here.sum())
            joint["symbols_ok"] += int((agree & here).sum())
            joint["usable_ok"] += int((usable & here).sum())
            staff_misreads += (((~usable) & here).sum(dim=1)).tolist()
            joint["notes"] += int((sounds & here).sum())
            joint["notes_ok"] += int((agree & sounds & here).sum())
            joint["notes_audible_ok"] += int((audible & sounds & here).sum())
            staff_errors += (((~agree) & here).sum(dim=1)).tolist()

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

    lift_names = names["lift"]
    if confusion:
        eprint("\n=== accidentals: what is mistaken for what ===")
        for (actual, predicted), count in confusion.most_common(15):
            eprint(
                f"   {lift_names.get(actual, actual):<12} read as "
                f"{lift_names.get(predicted, predicted):<12} {count:>6}"
            )

    # Symbols dropped or invented, the measure that matters most here.
    empty_index = config.lift_vocab.get("_")
    # Only rows that can carry an accidental count. A barline's lift column
    # holds the not-applicable mark, and reading that as blank is not a lost
    # accidental -- counting it as one hid the real figure behind the number of
    # barlines, which grew when the repeat signs came back.
    accidental = {
        index for index in truth["lift"] if lift_is_symbol(lift_names.get(index, ""))
    }
    dropped = sum(
        count for (actual, predicted), count in confusion.items()
        if predicted == empty_index and actual in accidental
    )
    invented = sum(
        count for (actual, predicted), count in confusion.items()
        if actual == empty_index and predicted in accidental
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

    weak = sorted(
        {
            name
            for (name, _), seen in by_pitch.items()
            if sum(v for (n, _), v in by_pitch.items() if n == name)
            and sum(v for (n, _), v in by_pitch_hit.items() if n == name)
            / sum(v for (n, _), v in by_pitch.items() if n == name)
            < 0.95
        }
    )
    if weak:
        eprint("\n=== the weak accidentals, split by the note underneath ===")
        for name in weak:
            eprint(f"   {name}")
            rows = sorted(
                ((pitch, seen) for (n, pitch), seen in by_pitch.items() if n == name),
                key=lambda item: -item[1],
            )
            for pitch, seen in rows:
                got = by_pitch_hit[(name, pitch)]
                eprint(f"      {pitch:<5}{got:>5}/{seen:<5}{100 * got / seen:>7.1f}%")

    if joint["symbols"]:
        perfect = sum(1 for count in staff_errors if count == 0)
        near = sum(1 for count in staff_errors if count <= 1)
        errors = sum(staff_errors)
        eprint("\n=== everything right at once ===")
        eprint(
            f"   notes with duration, pitch and accidental all correct : "
            f"{joint['notes_ok']}/{joint['notes']} = "
            f"{100 * joint['notes_ok'] / joint['notes']:.1f}%"
        )
        eprint(
            f"   notes a listener would hear correctly (pitch+accidental): "
            f"{joint['notes_audible_ok']}/{joint['notes']} = "
            f"{100 * joint['notes_audible_ok'] / joint['notes']:.1f}%"
        )
        eprint(
            f"   symbols of every kind fully correct                    : "
            f"{joint['symbols_ok']}/{joint['symbols']} = "
            f"{100 * joint['symbols_ok'] / joint['symbols']:.1f}%"
        )
        eprint(
            f"   staffs with no mistake at all                          : "
            f"{perfect}/{len(staff_errors)} = "
            f"{100 * perfect / len(staff_errors):.1f}%"
        )
        eprint(
            f"   staffs with at most one mistake                        : "
            f"{near}/{len(staff_errors)} = "
            f"{100 * near / len(staff_errors):.1f}%"
        )
        eprint(
            f"   mistakes per staff, on average                         : "
            f"{errors / len(staff_errors):.2f}"
        )
        clean = sum(1 for count in staff_misreads if count == 0)
        eprint(
            f"\n   ignoring the not-applicable marker on barlines and repeats:"
        )
        eprint(
            f"   symbols read correctly                                 : "
            f"{joint['usable_ok']}/{joint['symbols']} = "
            f"{100 * joint['usable_ok'] / joint['symbols']:.1f}%"
        )
        eprint(
            f"   staffs with no mistake at all                          : "
            f"{clean}/{len(staff_misreads)} = "
            f"{100 * clean / len(staff_misreads):.1f}%"
        )
        eprint(
            f"   mistakes per staff, on average                         : "
            f"{sum(staff_misreads) / len(staff_misreads):.2f}"
        )



def _align(
    reference: list[tuple[str, ...]], hypothesis: list[tuple[str, ...]]
) -> list[tuple[tuple[str, ...] | None, tuple[str, ...] | None]]:
    """Line two symbol sequences up by edit distance.

    Free reading can add or lose symbols, so position i of one sequence is not
    position i of the other. Pairs come back as (reference, hypothesis), with
    None on one side for a symbol dropped or invented.
    """
    rows, columns = len(reference), len(hypothesis)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        cost[row][0] = row
    for column in range(columns + 1):
        cost[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            substitute = cost[row - 1][column - 1] + (
                reference[row - 1] != hypothesis[column - 1]
            )
            cost[row][column] = min(
                substitute, cost[row - 1][column] + 1, cost[row][column - 1] + 1
            )

    pairs: list[tuple[tuple[str, ...] | None, tuple[str, ...] | None]] = []
    row, column = rows, columns
    while row or column:
        if row and column and cost[row][column] == cost[row - 1][column - 1] + (
            reference[row - 1] != hypothesis[column - 1]
        ):
            pairs.append((reference[row - 1], hypothesis[column - 1]))
            row, column = row - 1, column - 1
        elif row and cost[row][column] == cost[row - 1][column] + 1:
            pairs.append((reference[row - 1], None))
            row -= 1
        else:
            pairs.append((None, hypothesis[column - 1]))
            column -= 1
    pairs.reverse()
    return pairs


def _symbols_from_batch(
    batch: dict, names: dict[str, dict[int, str]], row: int
) -> list[tuple[str, ...]]:
    """The true symbol sequence for one staff, read back out of its labels."""
    keys = {
        "rhythm": "rhythms", "pitch": "pitchs", "lift": "lifts",
        "position": "positions", "articulations": "articulations",
    }
    length = int(batch["mask"][row].sum())
    symbols = []
    for step in range(length):
        rhythm_index = int(batch[keys["rhythm"]][row][step])
        rhythm = names["rhythm"].get(rhythm_index, "?")
        if rhythm in ("BOS", "EOS", "PAD"):
            continue
        symbols.append(
            tuple(
                names[branch].get(int(batch[keys[branch]][row][step]), "?")
                for branch in BRANCHES
            )
        )
    return symbols


def evaluate_generated(
    checkpoint: str | None, limit: int | None, index: str | None = None
) -> None:
    """Score the model reading each staff on its own, with nothing fed back."""
    config = Config()
    if checkpoint:
        config.filepaths.checkpoint = checkpoint
    if not os.path.exists(config.filepaths.checkpoint):
        eprint(f"No checkpoint at {config.filepaths.checkpoint}")
        sys.exit(1)

    chosen = index or symbtr_test_index
    if not os.path.exists(chosen):
        eprint(f"No test index at {chosen}. Unpack the dataset first.")
        sys.exit(1)

    with open(chosen, encoding="utf-8") as handle:
        samples = [line for line in handle if line.strip()]
    if limit:
        samples = samples[:limit]
    eprint(f"Reading {len(samples)} staffs with nothing fed back")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config)
    model.eval_mode()
    names = {
        branch: {index: token for token, index in vocab.items()}
        for branch, vocab in _vocabularies(config).items()
    }

    data = load_dataset(samples, config, validation_samples=samples)
    loader = torch.utils.data.DataLoader(data["validation"], batch_size=1, shuffle=False)

    total = substitutions = deletions = insertions = 0
    notes = notes_ok = usable_ok = 0
    staff_misreads: list[int] = []
    accidental_truth: collections.Counter = collections.Counter()
    accidental_hit: collections.Counter = collections.Counter()
    accidental_said: collections.Counter = collections.Counter()
    clean_staffs = 0
    staffs = 0

    with torch.no_grad():
        for batch in loader:
            image = batch["inputs"].to(device)
            produced = model.generate(image)
            hypothesis = [
                tuple(getattr(s, SYMBOL_ATTRIBUTE[branch]) for branch in BRANCHES)
                for s in produced
                if not s.is_control_symbol()
            ]
            reference = _symbols_from_batch(batch, names, 0)
            pairs = _align(reference, hypothesis)

            staffs += 1
            mistakes = 0
            misreads = 0
            for actual, predicted in pairs:
                if actual is None:
                    insertions += 1
                    mistakes += 1
                    misreads += 1
                    continue
                total += 1
                if predicted is None:
                    deletions += 1
                    mistakes += 1
                    misreads += 1
                elif actual != predicted:
                    substitutions += 1
                    mistakes += 1
                    # A barline or repeat sign fills its other four columns with
                    # the not-applicable marker. Getting the symbol right and
                    # the marker wrong is a matter of form, and nothing
                    # downstream reads those columns.
                    if has_rhythm_symbol_a_position(actual[0]) or actual[0] != predicted[0]:
                        misreads += 1
                    else:
                        usable_ok += 1
                else:
                    usable_ok += 1
                if actual[0].startswith("note"):
                    notes += 1
                    if actual == predicted:
                        notes_ok += 1
                if lift_is_symbol(actual[2]):
                    accidental_truth[actual[2]] += 1
                    if predicted is not None and predicted[2] == actual[2]:
                        accidental_hit[actual[2]] += 1
                if predicted is not None and lift_is_symbol(predicted[2]):
                    accidental_said[predicted[2]] += 1
            if not mistakes:
                clean_staffs += 1
            staff_misreads.append(misreads)

    if not total:
        eprint("Nothing to score")
        return

    errors = substitutions + deletions + insertions
    eprint("\n=== reading on its own, lined up by edit distance ===")
    eprint(f"   symbols in the test set : {total}")
    eprint(f"   wrong  (substitutions)  : {substitutions}")
    eprint(f"   dropped (deletions)     : {deletions}")
    eprint(f"   invented (insertions)   : {insertions}")
    eprint(f"   symbol error rate       : {100 * errors / total:.1f}%")
    eprint(f"   symbols read correctly  : {100 * (total - substitutions - deletions) / total:.1f}%")
    if notes:
        eprint(f"   notes fully correct     : {notes_ok}/{notes} = {100 * notes_ok / notes:.1f}%")
    eprint(f"   staffs with no mistake  : {clean_staffs}/{staffs} = {100 * clean_staffs / staffs:.1f}%")

    clean = sum(1 for count in staff_misreads if count == 0)
    eprint("\n   ignoring the not-applicable marker on barlines and repeats:")
    eprint(f"   symbols read correctly  : {usable_ok}/{total} = {100 * usable_ok / total:.1f}%")
    eprint(f"   staffs with no mistake  : {clean}/{staffs} = {100 * clean / staffs:.1f}%")
    eprint(
        f"   mistakes per staff      : {sum(staff_misreads) / staffs:.2f}"
    )

    if accidental_truth:
        eprint("\n=== accidentals, reading on its own ===")
        eprint(f"{'symbol':<12}{'in test':>9}{'found':>8}{'recall':>9}{'precision':>11}")
        for name, seen in accidental_truth.most_common():
            got, said = accidental_hit[name], accidental_said[name]
            eprint(
                f"{name:<12}{seen:>9}{got:>8}{100 * got / seen:>8.1f}%"
                f"{(100 * got / said if said else 0.0):>10.1f}%"
            )
        seen_all = sum(accidental_truth.values())
        got_all = sum(accidental_hit.values())
        eprint(f"{'TOTAL':<12}{seen_all:>9}{got_all:>8}{100 * got_all / seen_all:>8.1f}%")

def lift_is_symbol(name: str) -> bool:
    return name.startswith(("sharp", "flat")) or name == "N"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Model to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Use only N staffs.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--index",
        default=None,
        help="Score this list of staffs instead of the usual test split.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Let the model read each staff on its own instead of feeding it the "
             "correct previous symbols, and line the two sequences up by edit distance.",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Count a Western sharp as sharp4 and a Western flat as flat5, so a "
             "checkpoint without makam tokens can be scored fairly.",
    )
    options = parser.parse_args()
    if options.generate:
        evaluate_generated(options.checkpoint, options.limit, options.index)
    else:
        evaluate(
            options.checkpoint,
            options.limit,
            options.batch_size,
            options.lenient,
            options.index,
        )
