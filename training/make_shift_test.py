"""Build a test that asks whether the model reads a sign wherever it sits.

The ordinary test can only measure what it happens to contain. If a sign sits on
one degree in the training data and on the same degree in the test data, the
model can pass by learning the height rather than the shape, and nothing shows
it: the five-comma sharp only gave itself away because the split happened to put
thirteen of its rare F4 cases on the test side.

So the staffs are re-engraved a few degrees up or down and the model is asked to
read them again. Same music, same engraving, same labels except that every pitch
moves with the picture. Whatever the score drops between the two is what it was
reading from height rather than from shape.

    python -m training.make_shift_test --steps -2 --limit 300

Writes a folder of staff images with a matching index, ready to be evaluated
like any other test set.
"""

import argparse
import collections
import os
import re

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol
from training.datasets.convert_symbtr import git_root, symbtr_test_index
from training.datasets.shift_staff import leaves_the_staff, render_shifted, shift_tokens

STAFF = re.compile(r"-(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=-2, help="Degrees to move, - is down.")
    parser.add_argument("--limit", type=int, default=None, help="At most this many staffs.")
    parser.add_argument(
        "--out",
        default=os.path.join(git_root, "datasets", "SymbTr-shifted"),
        help="Where the images and the index go.",
    )
    options = parser.parse_args()
    os.makedirs(options.out, exist_ok=True)

    with open(symbtr_test_index, encoding="utf-8") as handle:
        samples = [line.strip() for line in handle if line.strip()]

    # A staff's number in its work is its number on the page, which is what the
    # renderer needs to find it again.
    made, skipped = 0, collections.Counter()
    index_lines = []
    for sample in samples:
        if options.limit and made >= options.limit:
            break
        image, tokens = sample.split(",")
        name = os.path.basename(image)[: -len(".png")]
        match = STAFF.search(name)
        if not match:
            skipped["odd name"] += 1
            continue
        stem, position = name[: match.start()], int(match.group(1))

        rows = []
        with open(tokens, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 5:
                    rows.append(EncodedSymbol(*parts))
        if not rows:
            skipped["no tokens"] += 1
            continue
        if leaves_the_staff(rows, options.steps):
            skipped["a note sits off the staff, so ledger lines would be wrong"] += 1
            continue
        moved = shift_tokens(rows, options.steps)
        if moved is None:
            skipped["pitch runs off the keyboard"] += 1
            continue

        target = os.path.join(options.out, f"{name}-s{options.steps:+d}.png")
        try:
            drawn = render_shifted(stem, position, options.steps, target)
        except Exception as error:  # noqa: BLE001
            skipped[type(error).__name__] += 1
            continue
        if not drawn:
            skipped["could not re-engrave"] += 1
            continue

        token_path = target[: -len(".png")] + ".tokens"
        with open(token_path, "w", encoding="utf-8") as handle:
            for symbol in moved:
                handle.write(str(symbol) + "\n")
        index_lines.append(
            f"{os.path.relpath(target, git_root)},{os.path.relpath(token_path, git_root)}\n".replace(
                os.sep, "/"
            )
        )
        made += 1

    index_path = os.path.join(options.out, f"index_shift{options.steps:+d}.txt")
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.writelines(index_lines)

    eprint(f"re-engraved {made} staffs, moved {options.steps:+d} degrees")
    eprint(f"index: {index_path}")
    for reason, count in skipped.most_common():
        eprint(f"  skipped {count}: {reason}")


if __name__ == "__main__":
    main()
