"""Tests verifying that docs/PARAMETERS.md rows match code attributes at HEAD."""

import ast
import os
import re
import pytest

import mtah.v2.deadball as deadball
import mtah.v2.decode as decode
import mtah.v2.fallback as fallback
import mtah.v2.touch as touch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = os.path.join(ROOT, "docs", "PARAMETERS.md")

MODULES = {
    "mtah/v2/touch.py": touch,
    "mtah/v2/deadball.py": deadball,
    "mtah/v2/decode.py": decode,
    "mtah/v2/fallback.py": fallback,
}


def parse_doc_rows(doc_path=DOC_PATH):
    """Parse table rows with columns (Name, Value, Where, What it does, Basis)."""
    assert os.path.exists(doc_path), f"Missing documentation file: {doc_path}"
    with open(doc_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    in_table = False
    header_indices = {}

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue

        temp = stripped.replace("\\|", "___ESCAPED_PIPE___")
        cells = [c.strip().replace("___ESCAPED_PIPE___", "|") for c in temp.split("|")[1:-1]]

        # Detect parameter table header (Name, Value, Where)
        if "Name" in cells and "Value" in cells and any("Where" in c for c in cells):
            in_table = True
            header_indices = {c.split("(")[0].strip(): idx for idx, c in enumerate(cells)}
            continue

        # Skip divider
        if in_table and all(re.match(r"^:?-+:?$", c) for c in cells):
            continue

        if in_table and len(cells) >= 5:
            name_idx = header_indices.get("Name", 0)
            val_idx = header_indices.get("Value", 1)
            where_idx = header_indices.get("Where", 2)
            what_idx = header_indices.get("What it does", 3)
            basis_idx = header_indices.get("Basis", 4)

            name = cells[name_idx].replace("`", "").strip()
            val_str = cells[val_idx].replace("`", "").strip()
            where_str = cells[where_idx].replace("`", "").strip()
            what_str = cells[what_idx].strip()
            basis_str = cells[basis_idx].strip()

            rows.append({
                "name": name,
                "value_str": val_str,
                "where": where_str,
                "what": what_str,
                "basis": basis_str,
            })

    return rows


def parse_numeric_value(val_str):
    """Parse string representation into Python numeric / tuple value."""
    clean = val_str.strip().replace("`", "")
    try:
        return ast.literal_eval(clean)
    except Exception:
        if "**" in clean:
            base, exp = clean.split("**")
            return int(base.strip()) ** int(exp.strip())
        raise ValueError(f"Could not parse numeric value: {val_str}")


def check_parameter_row(row):
    """Verify that a single documentation row matches the module attribute at import."""
    loc = row["where"]
    if ":" not in loc:
        return False
    file_path, line_str = loc.split(":", 1)
    file_path = file_path.strip()

    if file_path not in MODULES:
        return False

    mod = MODULES[file_path]
    name = row["name"]
    expected_val = parse_numeric_value(row["value_str"])

    assert hasattr(mod, name), f"Module {mod.__name__} has no attribute {name}"
    actual_val = getattr(mod, name)

    if isinstance(expected_val, float):
        assert isinstance(actual_val, (int, float)), f"{name}: expected float, got {type(actual_val)}"
        assert actual_val == pytest.approx(expected_val, rel=1e-6), (
            f"{mod.__name__}.{name}: expected {expected_val}, got {actual_val}"
        )
    elif isinstance(expected_val, tuple):
        assert isinstance(actual_val, tuple), f"{name}: expected tuple, got {type(actual_val)}"
        assert len(actual_val) == len(expected_val)
        for act, exp in zip(actual_val, expected_val):
            assert act == pytest.approx(exp, rel=1e-6)
    else:
        assert actual_val == expected_val, (
            f"{mod.__name__}.{name}: expected {expected_val!r}, got {actual_val!r}"
        )

    # Check line number at HEAD
    abs_file = os.path.join(ROOT, file_path)
    with open(abs_file, "r", encoding="utf-8") as f:
        src_lines = f.readlines()
    line_no = int(re.split(r"[,-]", line_str)[0].strip())
    assert line_no <= len(src_lines), f"Line {line_no} beyond EOF in {file_path}"
    assert name in src_lines[line_no - 1], (
        f"{file_path}:{line_no} does not contain {name!r}: {src_lines[line_no - 1]}"
    )

    return True


def test_parameters_doc_matches_code():
    """Verify each (name, value) against the module attribute at import."""
    rows = parse_doc_rows()
    assert len(rows) > 0, "No parameter rows parsed from docs/PARAMETERS.md"

    checked = 0
    for row in rows:
        loc = row["where"]
        fpath = loc.split(":")[0].strip() if ":" in loc else ""
        if fpath in MODULES:
            matched = check_parameter_row(row)
            if matched:
                checked += 1

    # Verify that all 62 decoder parameters across the 4 modules were checked
    assert checked >= 62, f"Expected 62 decoder parameters checked, got {checked}"


def test_parameters_doc_fails_on_mutation():
    """Verify that changing any parameter value causes verification to fail."""
    rows = parse_doc_rows()
    decoder_rows = [r for r in rows if (r["where"].split(":")[0].strip() if ":" in r["where"] else "") in MODULES]
    assert len(decoder_rows) > 0

    target = dict(decoder_rows[0])
    orig_val = parse_numeric_value(target["value_str"])
    if isinstance(orig_val, (int, float)):
        target["value_str"] = str(orig_val + 999.0)
    elif isinstance(orig_val, tuple):
        target["value_str"] = str(tuple(x + 999 for x in orig_val))

    with pytest.raises(AssertionError):
        check_parameter_row(target)


def test_basis_vocabulary():
    """Verify that Basis column strictly adheres to permitted honest vocabulary."""
    allowed_prefixes = (
        "official document (",
        "measured on the match without labels (",
        "football law (",
        "physical limit",
        "design choice, not validated against labels",
        "heuristic, no support found",
    )
    rows = parse_doc_rows()
    for row in rows:
        basis = row["basis"]
        valid = any(basis == p or basis.startswith(p) for p in allowed_prefixes)
        assert valid, f"Invalid basis vocabulary for {row['name']}: {basis!r}"
