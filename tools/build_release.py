"""Mechanically split the authorized final submission into readable modules.

This never executes the input source. It requires the recorded original hashes,
keeps the float32 buffer unchanged, and refuses to overwrite different files.
Run from the repository root: python -m tools.build_release /path/submission2.zip.
"""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zipfile

from tools.inspect_submission import ROOT, check_expected, extract_constants, inspect_archive, read_archive


def emit(path: Path, data: str | bytes) -> None:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite a different file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def binding_names(nodes: list[ast.stmt]) -> set[str]:
    names = set()
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return names


def imports_from(module: str, available: set[str], nodes: list[ast.stmt]) -> str:
    used = {n.id for node in nodes for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    selected = sorted(available & used)
    if not selected:
        return ""
    return f"from .{module} import (\n" + "".join(f"    {name},\n" for name in selected) + ")\n"


def render(nodes: list[ast.stmt], header: str, imports: str = "") -> str:
    return header + imports + "\n\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[])) + "\n"


class Rename(ast.NodeTransformer):
    mapping = {"_ArrayPolicyBase": "BasePolicy", "_StrategicArrayPolicy": "NumpyPolicy",
               "_TurnDecisionEngine": "DecisionController"}

    def visit_Name(self, node):
        node.id = self.mapping.get(node.id, node.id)
        return node

    def visit_ClassDef(self, node):
        old = node.name
        node.name = self.mapping.get(old, old)
        if old == "_ArrayPolicyBase":
            # This older forward method is shadowed by the deployed subclass.
            node.body = [n for n in node.body
                         if not (isinstance(n, ast.FunctionDef) and n.name == "scores_and_count")]
        return self.generic_visit(node)


def build(archive: Path, output: Path) -> dict:
    report = inspect_archive(archive)
    recorded = json.loads((ROOT / "evidence/submission_manifest.json").read_text())
    check_expected(report, recorded)
    files = read_archive(archive)
    source = files["main.py"].decode("utf-8")
    constants = extract_constants(files["main.py"])
    nodes = ast.parse(source).body
    start_weights = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "_PARAMETER_LAYOUT" for t in n.targets))
    restore = next(i for i, n in enumerate(nodes) if isinstance(n, ast.FunctionDef)
                   and n.name == "_restore_parameter_map")
    activation = next(i for i, n in enumerate(nodes) if isinstance(n, ast.FunctionDef)
                      and n.name == "_smooth_activation")
    schema = nodes[:start_weights]
    feature_nodes = nodes[restore + 1:activation]
    model_names = {"_smooth_activation", "_dense_layer", "_normalize_layer", "_probability_vector",
                   "_ArrayPolicyBase", "_StrategicArrayPolicy"}
    model_nodes = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in model_names]
    controller = [n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "_TurnDecisionEngine"][-1]
    mask_node = next(n for n in nodes if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_DEPLOYED_FEATURE_MASK" for t in n.targets))
    common = "from dataclasses import dataclass, field\nfrom collections import defaultdict\nfrom typing import Any\nimport hashlib\nimport math\nimport numpy as np\n"
    header = ("# Extracted from the preserved GALEX submission by tools/build_release.py.\n"
              "# Inference calculations are retained; see docs/RELEASE.md and NOTICE.md.\n")
    package = output / "ptcg_agent"
    emit(package / "schema.py", render(schema, header))
    emit(package / "features.py", render(feature_nodes, header,
         common + imports_from("schema", binding_names(schema), feature_nodes)))
    renamed_model = [Rename().visit(n) for n in model_nodes]
    emit(package / "model.py", render(renamed_model, header,
         "import numpy as np\nfrom .weights import load_parameters as _restore_parameter_map\n"))
    controller_nodes = [mask_node, Rename().visit(controller)]
    controller_imports = ("from pathlib import Path\nimport numpy as np\n"
                          "from .model import NumpyPolicy, _probability_vector\n"
                          + imports_from("schema", binding_names(schema), controller_nodes)
                          + imports_from("features", binding_names(feature_nodes), controller_nodes))
    emit(package / "controller.py", render(controller_nodes, header, controller_imports))
    payload = base64.b64decode(b"".join(constants["_PARAMETER_B64_CHUNKS"]), validate=True)
    layout = {
        "format": "little-endian-float32",
        "stored_float32_values": constants["_PARAMETER_FLOAT_COUNT"],
        "sha256": constants["_PARAMETER_SHA256"],
        "tensor_layout": constants["_PARAMETER_LAYOUT"],
        "source_main_sha256": report["files"]["main.py"]["sha256"],
        "deck_index": 0,
    }
    emit(package / "assets/parameters.f32", payload)
    emit(package / "assets/layout.json", json.dumps(layout, indent=2) + "\n")
    emit(package / "assets/deck.csv", files["deck.csv"])
    generated = ["schema.py", "features.py", "model.py", "controller.py",
                 "assets/parameters.f32", "assets/layout.json", "assets/deck.csv"]
    manifest = {
        "source_main_sha256": report["files"]["main.py"]["sha256"],
        "source_deck_sha256": report["files"]["deck.csv"]["sha256"],
        "parameter_buffer_unchanged": True,
        "transformations": ["Split schema, features, model and final controller into modules",
                            "Externalize unchanged float32 bytes and tensor layout",
                            "Rename runtime classes for readability",
                            "Omit shadowed earlier controller and unused base forward method",
                            "Keep feature definitions, including original later-name overrides"],
        "generated_files": {"ptcg_agent/" + name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                            for name in generated},
    }
    emit(output / "evidence/release_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def training_references(v67: Path, v76: Path, output: Path) -> None:
    selected = {v67: ["v67/model_v67.py", "v67/build_plan_targets.py", "v67/train_plan_bc.py"],
                v76: ["v76/build_dataset_latest.py", "v76/train_plan_scratch_all.py"]}
    manifest = {"scope": "Reference code only; full training dependencies and checkpoints are absent", "files": {}}
    for archive, members in selected.items():
        with zipfile.ZipFile(archive) as bundle:
            for member in members:
                original = bundle.read(member)
                body = original.decode("utf-8").replace('Path(r"E:\\PythonProject\\Pokemon")',
                                                        "Path(__file__).resolve().parents[1]")
                body = ("# Authorized reference implementation; not an end-to-end runnable training release.\n"
                        "# See training_reference/README.md and NOTICE.md.\n" + body)
                relative = "training_reference/" + member
                emit(output / relative, body)
                manifest["files"][relative] = {
                    "original_sha256": hashlib.sha256(original).hexdigest(),
                    "published_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "changes": "Add reference-status comments; replace machine-specific root where present",
                }
            if archive == v76:
                report = json.loads(bundle.read("v76/artifacts/dataset_report.json"))
                keys = ("episodes_scanned", "decisions", "option_rows", "target_deck",
                        "target_deck_index", "target_deck_decisions", "unique_decks", "decisions_by_day")
                safe_report = {k: report[k] for k in keys}
                safe_report["scope"] = "Reported prepared dataset, not regenerated; team identities and local paths omitted"
                emit(output / "training_reference/dataset_summary.json", json.dumps(safe_report, indent=2) + "\n")
    emit(output / "training_reference/source_manifest.json", json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--v67", type=Path)
    parser.add_argument("--v76", type=Path)
    args = parser.parse_args()
    if bool(args.v67) != bool(args.v76):
        parser.error("Provide both --v67 and --v76, or neither")
    manifest = build(args.archive, args.output)
    if args.v67:
        training_references(args.v67, args.v76, args.output)
    print(json.dumps({"generated_files": len(manifest["generated_files"]),
                      "parameter_buffer_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
