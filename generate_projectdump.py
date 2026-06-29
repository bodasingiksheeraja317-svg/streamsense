"""
generate_projectdump.py
Run from the STREAMSENSE repo root:
    python generate_projectdump.py
Outputs: projectdump.md
"""

import os
import json
import pathlib

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT = pathlib.Path(__file__).parent.resolve()
OUTPUT_FILE = REPO_ROOT / "projectdump.md"

# Directories whose CONTENTS are collapsed (folder shown, children skipped).
# Supports glob-style prefix matching via startswith on the relative path.
COLLAPSE_DIR_PREFIXES = (
    "data/",                        # all subfolders under data/
    "golden_vectors/labels",
    "golden_vectors/mel",
    "golden_vectors/normalized",
    "golden_vectors/raw",
    "golden_vectors/wav",
    "golden_vectors_10_matlab/labels",
    "golden_vectors_10_matlab/mel",
    "golden_vectors_10_matlab/normalized",
    "golden_vectors_10_matlab/raw",
    "golden_vectors_1000/labels",
    "golden_vectors_1000/mel",
    "golden_vectors_1000/normalized",
    "golden_vectors_1000/raw",
    "recordings",
    "unknown_data",
    "streamsense-env-win",
    "training/logs",
    "training/__pycache__",
    "checkpoints",
    "checkpoints_1d",
)

# Files to hide from the tree entirely
TREE_SKIP_FILES = {
    "generate_projectdump.py",
    "projectdump.md",
}

# Individual files to skip entirely (relative to repo root)
SKIP_FILES = {
    "golden_vectors/manifest.json",
    "golden_vectors_10_matlab/manifest.json",
    "golden_vectors_1000/manifest.json",
}

# Extensions that are always skipped as individual files
SKIP_EXTENSIONS = {".pth", ".onnx", ".qonnx", ".bin", ".npy", ".so", ".pyd"}

# Binary/bulk-data extensions — if a folder contains ONLY these, collapse it
BULK_EXTENSIONS = {".wav", ".txt", ".png", ".bin", ".npy", ".pth", ".onnx",
                   ".qonnx", ".so", ".pyd", ".csv", ".lab", ".TextGrid"}

# Files explicitly included (relative paths from repo root, training/ scripts etc.)
# If None, include everything not excluded.
INCLUDE_TRAINING_EXTENSIONS = {".py", ".m", ".json"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def rel(path: pathlib.Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def should_collapse_dir(path: pathlib.Path) -> bool:
    r = rel(path)
    # Exact match or prefix match
    for prefix in COLLAPSE_DIR_PREFIXES:
        if r == prefix.rstrip("/") or r.startswith(prefix.rstrip("/") + "/") or r.startswith(prefix):
            return True
    # Auto-collapse: folder whose direct children are all bulk extensions
    # But never collapse evaluation dirs even if they only contain .txt/.png
    if r.startswith("evaluation"):
        return False
    try:
        children = list(path.iterdir())
        if children and all(
            (c.is_file() and c.suffix.lower() in BULK_EXTENSIONS) or
            (c.is_dir() and c.name in {"__pycache__"})
            for c in children
        ):
            return True
    except PermissionError:
        pass
    return False


def should_skip_file_in_tree(path: pathlib.Path) -> bool:
    if path.name in TREE_SKIP_FILES:
        return True
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True
    r = rel(path)
    if r in SKIP_FILES:
        return True
    return False


# ── Tree builder ──────────────────────────────────────────────────────────────

def build_tree(root: pathlib.Path) -> list[str]:
    lines = []

    def _walk(path: pathlib.Path, prefix: str):
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        # Filter out hidden git internals from display but keep .gitattributes/.gitignore
        entries = [e for e in entries if not (e.name.startswith(".git") and e.is_dir())]

        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            extension = "    " if i == len(entries) - 1 else "│   "

            if entry.is_dir():
                if should_collapse_dir(entry):
                    lines.append(f"{prefix}{connector}{entry.name}/ [...]")
                else:
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    _walk(entry, prefix + extension)
            else:
                if not should_skip_file_in_tree(entry):
                    lines.append(f"{prefix}{connector}{entry.name}")

    lines.append(f"{root.name}/")
    _walk(root, "")
    return lines


# ── Content collectors ────────────────────────────────────────────────────────

def collect_python_files() -> list[pathlib.Path]:
    collected = []
    search_dirs = [
        REPO_ROOT / "training",
        REPO_ROOT,          # root-level .py if any
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix in {".py", ".m"} and f.name != "generate_projectdump.py":
                collected.append(f)
    return collected


def collect_json_files() -> list[pathlib.Path]:
    collected = []
    candidates = [
        REPO_ROOT / "stats" / "normalization_stats.json",
        REPO_ROOT / "stats" / "golden_selection.json",
        REPO_ROOT / "class_labels.json",
    ]
    for f in candidates:
        if f.exists():
            collected.append(f)
    return collected


def collect_eval_reports() -> list[pathlib.Path]:
    collected = []
    for d in [REPO_ROOT / "evaluation", REPO_ROOT / "evaluation_1d"]:
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix == ".txt":
                    collected.append(f)
    return collected


def collect_notebooks() -> list[pathlib.Path]:
    collected = []
    # Root-level notebooks
    for f in sorted(REPO_ROOT.iterdir()):
        if f.is_file() and f.suffix == ".ipynb":
            collected.append(f)
    # training/ notebooks
    td = REPO_ROOT / "training"
    if td.exists():
        for f in sorted(td.iterdir()):
            if f.is_file() and f.suffix == ".ipynb":
                collected.append(f)
    # onnx_models/ notebooks
    od = REPO_ROOT / "onnx_models"
    if od.exists():
        for f in sorted(od.iterdir()):
            if f.is_file() and f.suffix == ".ipynb":
                collected.append(f)
    return collected


def strip_notebook_outputs(path: pathlib.Path) -> str:
    """Return notebook source with only input cells, outputs removed."""
    with open(path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)

    lines = []
    for cell in nb.get("cells", []):
        ct = cell.get("cell_type", "")
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if ct == "code":
            lines.append(f"# [code cell]\n{src}")
        elif ct == "markdown":
            lines.append(f"# [markdown]\n# " + src.replace("\n", "\n# "))
        lines.append("")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sections = []

    # ── 1. Project tree ──
    tree_lines = build_tree(REPO_ROOT)
    sections.append("# STREAMSENSE — Project Dump\n")
    sections.append("## Project Tree\n")
    sections.append("```")
    sections.extend(tree_lines)
    sections.append("```\n")

    # ── 2. Python / MATLAB source files ──
    sections.append("---\n## Source Files\n")
    for f in collect_python_files():
        r = rel(f)
        lang = "python" if f.suffix == ".py" else "matlab"
        sections.append(f"### `{r}`\n")
        sections.append(f"```{lang}")
        sections.append(f.read_text(encoding="utf-8", errors="replace"))
        sections.append("```\n")

    # ── 3. JSON config / stats ──
    sections.append("---\n## Config & Stats\n")
    for f in collect_json_files():
        r = rel(f)
        sections.append(f"### `{r}`\n")
        sections.append("```json")
        sections.append(f.read_text(encoding="utf-8", errors="replace"))
        sections.append("```\n")

    # ── 4. Evaluation reports ──
    sections.append("---\n## Evaluation Reports\n")
    for f in collect_eval_reports():
        r = rel(f)
        sections.append(f"### `{r}`\n")
        sections.append("```")
        sections.append(f.read_text(encoding="utf-8", errors="replace"))
        sections.append("```\n")

    # ── 5. Notebooks (input cells only) ──
    sections.append("---\n## Notebooks (input cells only)\n")
    for f in collect_notebooks():
        r = rel(f)
        sections.append(f"### `{r}`\n")
        sections.append("```python")
        sections.append(strip_notebook_outputs(f))
        sections.append("```\n")

    # ── Write output ──
    output = "\n".join(sections)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print(f"✓ projectdump.md written ({len(output):,} chars)")
    print(f"  Path: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
