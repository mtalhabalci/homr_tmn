"""Bundle the converted SymbTr staff samples for upload to Google Drive.

The staff images are far too large for git (~250 MB), so the Colab notebook
either unpacks this archive from Drive or regenerates everything from the SymbTr
sources. Unpacking is roughly fifteen times quicker.

    python -m notebooks.package_dataset

Writes symbtr_veri.tar.gz into the colab folder beside the repository, where
the run logs and evaluations also live. Paths inside the archive are
relative to the repository root, so extracting it into a fresh clone puts every
file exactly where the index files expect it.
"""

import argparse
import os
import tarfile
from pathlib import Path

from homr.simple_logging import eprint
from training.datasets.convert_symbtr import (
    index_file,
    index_test,
    index_train,
    index_val,
    working_dir,
)

git_root = Path(__file__).parent.parent.absolute()


def package(destination: str, working_dir: str = working_dir, also: list[str] | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    if not os.path.isdir(working_dir):
        eprint(f"Nothing to package: {working_dir} does not exist.")
        eprint("Run python -m training.datasets.convert_symbtr first.")
        raise SystemExit(1)

    staff_files = sorted(
        name for name in os.listdir(working_dir) if name.endswith((".png", ".tokens"))
    )
    if not staff_files:
        eprint(f"Nothing to package: no staff samples in {working_dir}")
        raise SystemExit(1)

    eprint(f"Packing {len(staff_files)} staff files into {destination}")
    with tarfile.open(destination, "w:gz") as archive:
        for name in staff_files:
            archive.add(
                os.path.join(working_dir, name),
                os.path.relpath(os.path.join(working_dir, name), git_root).replace(os.sep, "/"),
            )
        for index in (index_file, index_train, index_val, index_test):
            if os.path.exists(index):
                archive.add(
                    index, os.path.relpath(index, git_root).replace(os.sep, "/")
                )
        # Further folders whose staffs the index files point into, such as the
        # courtesy-accidental staffs, taken whole with their own index files.
        for folder in also or []:
            count = 0
            for root, _, names in os.walk(folder):
                for name in sorted(names):
                    if name.endswith((".png", ".tokens", ".txt")):
                        path = os.path.join(root, name)
                        archive.add(path, os.path.relpath(path, git_root).replace(os.sep, "/"))
                        count += 1
            eprint(f"  and {count} files from {folder}")
    size = os.path.getsize(destination) / 1e6
    eprint(f"Wrote {destination} ({size:.0f} MB)")
    eprint("Upload it to a Google Drive folder named homr_makam.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(git_root.parent / "colab" / "symbtr_veri.tar.gz"),
        help="Where to write the archive.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Folder under datasets/ holding the staff files, if not SymbTr-work.",
    )
    parser.add_argument(
        "--also",
        action="append",
        default=[],
        help="Another folder under datasets/ to include whole. May be repeated.",
    )
    options = parser.parse_args()
    folder = os.path.join(git_root, "datasets", options.work_dir) if options.work_dir else working_dir
    package(options.out, folder, [os.path.join(git_root, "datasets", extra) for extra in options.also])
