"""Execute every supplied notebook from top to bottom."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).parents[1]
REFERENCE_NOTEBOOKS = sorted((PROJECT_ROOT / "examples").glob("**/*.ipynb"))
GUIDED_TEMPLATE = PROJECT_ROOT / "notebooks/00-guided-case-study-template.ipynb"
NOTEBOOKS = [*REFERENCE_NOTEBOOKS, GUIDED_TEMPLATE]


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_has_case_study_structure(notebook_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    markdown = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    source = "\n".join(cell.source for cell in code)

    assert len(markdown) >= 3
    assert len(code) >= 2
    assert markdown[0].source.startswith("# ")
    assert "dsm_project" in source


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_math_delimiters_are_balanced(notebook_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        lines = [line.strip() for line in cell.source.splitlines()]
        assert "[" not in lines, "Use $$ rather than a bare [ math delimiter"
        assert "]" not in lines, "Use $$ rather than a bare ] math delimiter"
        assert cell.source.count("$$") % 2 == 0


@pytest.mark.parametrize(
    "notebook_path", REFERENCE_NOTEBOOKS, ids=lambda path: path.name
)
def test_reference_notebook_explains_student_transfer(notebook_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    introduction = notebook.cells[0].source

    assert "**What this example models.**" in introduction
    assert "**Your project version.**" in introduction


def test_guided_template_contains_complete_applied_story() -> None:
    notebook = nbformat.read(GUIDED_TEMPLATE, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    required_sections = (
        "Investigation contract",
        "Data provenance and fitness for purpose",
        "Purposeful exploration",
        "Experimental boundaries and baseline",
        "Mathematical model",
        "Student implementation and verification",
        "Results, uncertainty, and practical meaning",
        "Failure analysis",
        "Conclusion and production decision",
        "Definition of done",
    )

    for section in required_sections:
        assert section in markdown
    assert "real-world or scientific question" in markdown


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_executes_from_a_clean_kernel(notebook_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="dsm-project",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )

    client.execute()
