"""Packaging checks: declared dependencies match the third-party imports in src/dsns."""
import ast
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "dsns"

# import name -> PyPI distribution name
IMPORT_TO_DIST = {
    "PIL": "pillow",
    "PyAstronomy": "PyAstronomy",
    "networkit": "networkit",
    "numpy": "numpy",
    "pyrender": "pyrender",
    "sgp4": "sgp4",
    "tqdm": "tqdm",
    "trimesh": "trimesh",
}
DOCS_DEPS = {"sphinx", "sphinx-argparse", "sphinx-rtd-theme"}


def _norm(req: str) -> str:
    name = re.split(r"[<>=!~;\[ ]", req.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def _pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _third_party_imports() -> set:
    found = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {m for m in found if m not in sys.stdlib_module_names and m != "dsns"}


def test_imports_are_mapped_and_declared():
    deps = {_norm(d) for d in _pyproject()["project"]["dependencies"]}
    imports = _third_party_imports()
    unmapped = imports - set(IMPORT_TO_DIST)
    assert not unmapped, f"add to IMPORT_TO_DIST: {unmapped}"
    missing = {m for m in imports if _norm(IMPORT_TO_DIST[m]) not in deps}
    assert not missing, f"imported but not declared: {missing}"



def test_declared_dependencies_are_imported():
    # Every runtime dependency should be used by the package; example-only tools belong elsewhere.
    deps = {_norm(d) for d in _pyproject()["project"]["dependencies"]}
    used = {_norm(IMPORT_TO_DIST[m]) for m in _third_party_imports() if m in IMPORT_TO_DIST}
    assert not (deps - used), f"declared but never imported: {deps - used}"

def test_docs_deps_only_in_docs_extra():
    project = _pyproject()["project"]
    deps = {_norm(d) for d in project["dependencies"]}
    assert not (deps & DOCS_DEPS)
    docs = {_norm(d) for d in project["optional-dependencies"]["docs"]}
    assert DOCS_DEPS <= docs


def test_no_junk_requirements():
    deps = _pyproject()["project"]["dependencies"]
    assert not any("python_version" in d for d in deps)
    assert len({_norm(d) for d in deps}) == len(deps)


def test_package_data_files_exist():
    pd = _pyproject()["tool"]["setuptools"]["package-data"]["dsns"]
    for pattern in pd:
        assert list(SRC.glob(pattern)), f"package-data pattern matches nothing: {pattern}"
