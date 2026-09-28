"""Keep the audience material runnable, linked, and in the workshop's house style."""

from pathlib import Path
import re
import unicodedata

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks" / "west").glob("*.py"))
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def _markdown_cells(path):
    cells, current = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# MAGIC"):
            current.append(line.removeprefix("# MAGIC").removeprefix(" ").removeprefix("%md").lstrip("\n"))
        elif current:
            cells.append("\n".join(current))
            current = []
    if current:
        cells.append("\n".join(current))
    return cells


def _prose(markdown):
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _slug(heading):
    heading = re.sub(r"[`*_]", "", heading.strip().lower())
    heading = "".join(ch for ch in heading if ch in "- " or unicodedata.category(ch)[0] in "LN")
    return heading.replace(" ", "-")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebooks_are_valid_databricks_source(path):
    source = path.read_text(encoding="utf-8")
    assert source.startswith("# Databricks notebook source\n")
    compile(source, str(path), "exec")
    for cell in source.split("# COMMAND ----------")[1:]:
        assert cell.strip(), f"{path.name} has an empty cell"


@pytest.mark.parametrize("path", [*DOCS, *NOTEBOOKS], ids=lambda path: path.name)
def test_prose_uses_no_semicolons_or_dashes(path):
    markdown = path.read_text(encoding="utf-8") if path.suffix == ".md" else "\n\n".join(_markdown_cells(path))
    for number, line in enumerate(_prose(markdown).splitlines(), start=1):
        for mark, name in ((";", "semicolon"), ("—", "em dash"), ("–", "en dash")):
            assert mark not in line, f"{path.name}: {name} in prose: {line.strip()[:120]}"


def test_readme_links_and_anchors_resolve():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    anchors = {_slug(match) for match in re.findall(r"(?m)^#{1,6} (.+)$", readme)}
    for target in re.findall(r"\]\(([^)\s]+)\)", readme):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = target.partition("#")
        if path:
            assert (ROOT / path).exists(), f"README links to a missing file: {target}"
        if anchor and not path:
            assert anchor in anchors, f"README links to a missing section: #{anchor}"


def test_every_notebook_is_listed_in_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for path in NOTEBOOKS:
        assert f"notebooks/west/{path.name}" in readme, path.name


def test_links_into_the_readme_from_notebooks_and_docs_resolve():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    anchors = {_slug(match) for match in re.findall(r"(?m)^#{1,6} (.+)$", readme)}
    for path in [*NOTEBOOKS, *DOCS[1:]]:
        text = path.read_text(encoding="utf-8")
        for anchor in re.findall(r"github\.com/debu-sinha/mlflow-eval-workshop(?:/blob/main/README\.md)?#([\w-]+)", text):
            assert anchor in anchors, f"{path.name} links to a missing README section: #{anchor}"
        for target in re.findall(r"\]\((\.\./[^)\s]+)\)", text):
            file_part, _, anchor = target.partition("#")
            assert (path.parent / file_part).resolve().exists(), f"{path.name} links to a missing file: {target}"
            if anchor and file_part.endswith("README.md"):
                assert anchor in anchors, f"{path.name} links to a missing README section: #{anchor}"
