#!/usr/bin/env python3
"""Optional end-to-end test for projection/reflection GGUF quantization.

This test needs an existing one-shard GGUF with a square attention-output
matrix. It is intentionally not registered with CTest because the repository
does not ship model weights. With --baseline-quantizer it also proves that the
current scale-1 payload is byte-identical to a pre-patch engine closure.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "gguf-py"))
from gguf import GGUFReader, GGUFWriter, dequantize  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_map(path):
    reader = GGUFReader(path, mode="r")
    return reader, {tensor.name: tensor for tensor in reader.tensors}


def payload_digest(tensor):
    return hashlib.sha256(memoryview(tensor.data)).hexdigest()


def verify_changed_payloads(source_path, source_reader, source_tensors, candidate_path, target):
    candidate_reader, candidate_tensors = tensor_map(candidate_path)
    if source_reader.data_offset != candidate_reader.data_offset:
        raise SystemExit("FAIL: GGUF header length changed")
    with source_path.open("rb") as left, candidate_path.open("rb") as right:
        if left.read(source_reader.data_offset) != right.read(candidate_reader.data_offset):
            raise SystemExit("FAIL: GGUF header bytes changed")
    if set(source_tensors) != set(candidate_tensors):
        raise SystemExit("FAIL: tensor names changed")
    changed = []
    for name, source_tensor in source_tensors.items():
        candidate_tensor = candidate_tensors[name]
        if (source_tensor.tensor_type, source_tensor.n_bytes) != (
                candidate_tensor.tensor_type, candidate_tensor.n_bytes):
            raise SystemExit(f"FAIL: tensor layout changed: {name}")
        if payload_digest(source_tensor) != payload_digest(candidate_tensor):
            changed.append(name)
    if changed != [target]:
        raise SystemExit(f"FAIL: changed payload set is {changed!r}")
    return candidate_reader, candidate_tensors


def quantizer_args(quantizer, direction, target, scale, patch_existing):
    result = [
        str(quantizer), "--allow-requantize", "--pure", "--keep-split",
        "--orthogonalize-control-vector", str(direction),
        "--orthogonalize-layer-range", "1", "2",
        "--orthogonalize-subspace-rank", "2",
        "--orthogonalize-pattern", f"^{re.escape(target)}$",
        "--orthogonalize-scale", str(scale),
        "--orthogonalize-expected-count", "1",
        "--orthogonalize-quant-passes", "16",
        "--orthogonalize-quant-correction", "0.25",
        "--orthogonalize-max-residual", "0.10",
    ]
    if patch_existing:
        result.insert(9, "--orthogonalize-patch-existing")
    return result


def write_direction(path, width):
    writer = GGUFWriter(path, "controlvector")
    writer.add_string("controlvector.model_hint", "patch-existing-test")
    writer.add_int32("controlvector.layer_count", 2)
    first = np.zeros(width, dtype=np.float32)
    second = np.zeros(width, dtype=np.float32)
    first[0] = 1.0
    second[1] = 1.0
    writer.add_tensor("direction.1", first)
    writer.add_tensor("direction.2", second)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("quantizer", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--target", default="blk.0.attn_output.weight")
    parser.add_argument("--baseline-quantizer", type=Path)
    parser.add_argument("--baseline-library-dir", type=Path)
    args = parser.parse_args()
    args.quantizer = args.quantizer.resolve()
    args.model = args.model.resolve()
    if not args.quantizer.is_file() or not os.access(args.quantizer, os.X_OK):
        parser.error("quantizer is not executable")
    if not args.model.is_file():
        parser.error("model is not readable")
    if bool(args.baseline_quantizer) != bool(args.baseline_library_dir):
        parser.error("--baseline-quantizer and --baseline-library-dir are required together")
    baseline_env = None
    if args.baseline_quantizer:
        args.baseline_quantizer = args.baseline_quantizer.resolve()
        args.baseline_library_dir = args.baseline_library_dir.resolve()
        if not args.baseline_quantizer.is_file() or not os.access(args.baseline_quantizer, os.X_OK):
            parser.error("baseline quantizer is not executable")
        if not args.baseline_library_dir.is_dir():
            parser.error("baseline library directory is not readable")
        baseline_env = os.environ.copy()
        old_library_path = baseline_env.get("LD_LIBRARY_PATH")
        baseline_env["LD_LIBRARY_PATH"] = str(args.baseline_library_dir) + (
            os.pathsep + old_library_path if old_library_path else "")

    source_reader, source_tensors = tensor_map(args.model)
    if args.target not in source_tensors:
        parser.error(f"model lacks target tensor {args.target}")
    target = source_tensors[args.target]
    shape = tuple(int(value) for value in target.shape)
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 2:
        parser.error(f"target must be square and at least width 2; got {shape}")
    width = shape[0]
    source_hash = sha256(args.model)

    with tempfile.TemporaryDirectory(
            prefix=".llama-patch-existing-", dir=args.model.parent) as directory_name:
        directory = Path(directory_name)
        direction = directory / "direction.gguf"
        write_direction(direction, width)

        def clone_candidate(stem):
            prefix = directory / stem
            path = directory / f"{stem}-00001-of-00001.gguf"
            subprocess.run([
                "cp", "--reflink=always", "--preserve=mode,timestamps", "--",
                str(args.model), str(path),
            ], check=True)
            path.chmod(path.stat().st_mode | 0o200)
            if (args.model.stat().st_dev, args.model.stat().st_ino) == (
                    path.stat().st_dev, path.stat().st_ino):
                raise SystemExit("FAIL: candidate is not a distinct inode")
            return prefix, path, sha256(path)

        output_prefix, candidate, pristine_candidate_hash = clone_candidate("candidate-scale1")
        common = quantizer_args(
            args.quantizer, direction, args.target, "1.0", patch_existing=True)
        invalid_correction = subprocess.run(
            common + ["--orthogonalize-quant-correction", "0",
                      str(args.model), str(output_prefix) + ".gguf", "Q8_0", "4"],
            capture_output=True, text=True)
        if (invalid_correction.returncode == 0 or
                "orthogonalize-quant-correction must be finite and in (0, 1]" not in
                invalid_correction.stdout + invalid_correction.stderr):
            raise SystemExit("FAIL: invalid correction fraction did not fail closed")
        if sha256(candidate) != pristine_candidate_hash:
            raise SystemExit("FAIL: invalid correction fraction changed the candidate")

        mismatch = subprocess.run(
            common + [str(args.model), str(output_prefix) + ".gguf", "Q5_K", "4"],
            capture_output=True, text=True)
        if mismatch.returncode == 0 or "patch-existing layout mismatch" not in (
                mismatch.stdout + mismatch.stderr):
            raise SystemExit("FAIL: incompatible output type did not fail closed")
        if sha256(candidate) != pristine_candidate_hash:
            raise SystemExit("FAIL: incompatible output type changed the candidate")

        dry = subprocess.run(
            common + ["--dry-run", str(args.model), str(output_prefix) + ".gguf", "Q8_0", "4"],
            capture_output=True, text=True)
        if dry.returncode != 0:
            raise SystemExit("FAIL: dry run failed:\n" + dry.stdout + dry.stderr)
        if "patch-existing validated" not in dry.stdout + dry.stderr:
            raise SystemExit("FAIL: dry run did not validate existing output")
        if sha256(candidate) != pristine_candidate_hash:
            raise SystemExit("FAIL: dry run changed the candidate")

        actual = subprocess.run(
            common + [str(args.model), str(output_prefix) + ".gguf", "Q8_0", "4"],
            capture_output=True, text=True)
        if actual.returncode != 0:
            raise SystemExit("FAIL: actual patch failed:\n" + actual.stdout + actual.stderr)
        log = actual.stdout + actual.stderr
        if log.count("patched-existing shard=") != 1:
            raise SystemExit("FAIL: actual run did not patch exactly one payload")
        if sha256(args.model) != source_hash:
            raise SystemExit("FAIL: source model changed")
        verify_changed_payloads(
            args.model, source_reader, source_tensors, candidate, args.target)

        if args.baseline_quantizer:
            baseline_prefix, baseline_candidate, _ = clone_candidate("candidate-scale1-baseline")
            baseline_common = quantizer_args(
                args.baseline_quantizer, direction, args.target, "1.0", patch_existing=True)
            baseline = subprocess.run(
                baseline_common + [str(args.model), str(baseline_prefix) + ".gguf", "Q8_0", "4"],
                capture_output=True, text=True, env=baseline_env)
            if baseline.returncode != 0:
                raise SystemExit("FAIL: baseline scale-1 patch failed:\n" + baseline.stdout + baseline.stderr)
            verify_changed_payloads(
                args.model, source_reader, source_tensors, baseline_candidate, args.target)
            if sha256(candidate) != sha256(baseline_candidate):
                raise SystemExit("FAIL: current scale-1 output is not byte-identical to baseline")
            residual_pattern = rf"orthogonalize: {re.escape(args.target)} post-quant-residual=([0-9.]+)%"
            current_residual = re.search(residual_pattern, log)
            baseline_residual = re.search(residual_pattern, baseline.stdout + baseline.stderr)
            if not current_residual or not baseline_residual or (
                    current_residual.group(1) != baseline_residual.group(1)):
                raise SystemExit("FAIL: current and baseline scale-1 residual logs differ")

        scale2_prefix, scale2_candidate, scale2_pristine_hash = clone_candidate("candidate-scale2")
        for invalid_scale in ("0", "-1", "2.0001", "nan", "inf"):
            invalid = subprocess.run(
                quantizer_args(args.quantizer, direction, args.target, invalid_scale, True) + [
                    str(args.model), str(scale2_prefix) + ".gguf", "Q8_0", "4"],
                capture_output=True, text=True)
            if invalid.returncode == 0 or "--orthogonalize-scale must be finite and in (0, 2]" not in (
                    invalid.stdout + invalid.stderr):
                raise SystemExit(f"FAIL: invalid scale {invalid_scale!r} did not fail closed")
            if sha256(scale2_candidate) != scale2_pristine_hash:
                raise SystemExit(f"FAIL: invalid scale {invalid_scale!r} changed the candidate")

        scale2_common = quantizer_args(
            args.quantizer, direction, args.target, "2.0", patch_existing=True)
        scale2_dry = subprocess.run(
            scale2_common + ["--dry-run", str(args.model), str(scale2_prefix) + ".gguf", "Q8_0", "4"],
            capture_output=True, text=True)
        if scale2_dry.returncode != 0 or "patch-existing validated" not in (
                scale2_dry.stdout + scale2_dry.stderr):
            raise SystemExit("FAIL: scale-2 dry run failed:\n" + scale2_dry.stdout + scale2_dry.stderr)
        if sha256(scale2_candidate) != scale2_pristine_hash:
            raise SystemExit("FAIL: scale-2 dry run changed the candidate")

        scale2 = subprocess.run(
            scale2_common + [str(args.model), str(scale2_prefix) + ".gguf", "Q8_0", "4"],
            capture_output=True, text=True)
        if scale2.returncode != 0:
            raise SystemExit("FAIL: scale-2 patch failed:\n" + scale2.stdout + scale2.stderr)
        scale2_log = scale2.stdout + scale2.stderr
        final_scale2 = re.search(
            rf"orthogonalize: {re.escape(args.target)} post-quant-residual=([0-9.]+)% "
            r"actual-source-component=([0-9.]+)% absolute-component=([0-9.]+)%",
            scale2_log)
        if ("target-error=" not in scale2_log or
                "post-quant target-relative error" not in scale2_log or
                not final_scale2):
            raise SystemExit("FAIL: scale-2 target-relative diagnostics are incomplete")
        if float(final_scale2.group(1)) > 10.0:
            raise SystemExit("FAIL: scale-2 target-relative error exceeds the fixture limit")
        if not 90.0 <= float(final_scale2.group(2)) <= 110.0:
            raise SystemExit("FAIL: scale-2 actual component does not retain reflected magnitude")
        _, scale2_tensors = verify_changed_payloads(
            args.model, source_reader, source_tensors, scale2_candidate, args.target)
        source_values = dequantize(
            np.asarray(source_tensors[args.target].data),
            source_tensors[args.target].tensor_type,
        )
        scale2_values = dequantize(
            np.asarray(scale2_tensors[args.target].data),
            scale2_tensors[args.target].tensor_type,
        )
        intended_values = source_values.copy()
        if args.target == "token_embd.weight":
            intended_values[:, :2] *= -1
            selected_error = scale2_values[:, :2] - intended_values[:, :2]
            selected_source = source_values[:, :2]
            orthogonal_error = scale2_values[:, 2:] - source_values[:, 2:]
            orthogonal_source = source_values[:, 2:]
        else:
            intended_values[:2, :] *= -1
            selected_error = scale2_values[:2, :] - intended_values[:2, :]
            selected_source = source_values[:2, :]
            orthogonal_error = scale2_values[2:, :] - source_values[2:, :]
            orthogonal_source = source_values[2:, :]
        external_target_error = np.linalg.norm(selected_error) / np.linalg.norm(selected_source)
        external_orthogonal_error = (
            np.linalg.norm(orthogonal_error) / np.linalg.norm(orthogonal_source)
            if orthogonal_source.size else 0.0
        )
        if external_target_error > 0.10:
            raise SystemExit("FAIL: independently decoded scale-2 target is not reflected")
        if external_orthogonal_error > 1e-6:
            raise SystemExit("FAIL: independently decoded orthogonal component changed")

        ordinary_prefix = directory / "ordinary-scale2"
        ordinary_common = quantizer_args(
            args.quantizer, direction, args.target, "2.0", patch_existing=False)
        ordinary_common.insert(3, "--keep-f32")
        ordinary = subprocess.run(
            ordinary_common + [str(args.model), str(ordinary_prefix) + ".gguf", "Q8_0", "4"],
            capture_output=True, text=True)
        if ordinary.returncode != 0:
            raise SystemExit("FAIL: ordinary scale-2 output failed:\n" + ordinary.stdout + ordinary.stderr)
        ordinary_paths = list(directory.glob("ordinary-scale2*.gguf"))
        if len(ordinary_paths) != 1:
            raise SystemExit(f"FAIL: ordinary scale-2 output set is {ordinary_paths!r}")
        _, ordinary_tensors = tensor_map(ordinary_paths[0])
        if args.target not in ordinary_tensors:
            raise SystemExit("FAIL: ordinary scale-2 output lost its target")
        if (ordinary_tensors[args.target].tensor_type,
                payload_digest(ordinary_tensors[args.target])) != (
                scale2_tensors[args.target].tensor_type,
                payload_digest(scale2_tensors[args.target])):
            raise SystemExit("FAIL: ordinary and patch-existing scale-2 target payloads differ")
        if "post-quant target-relative error" not in ordinary.stdout + ordinary.stderr:
            raise SystemExit("FAIL: ordinary scale-2 output lacks target-relative summary")

        if sha256(args.model) != source_hash:
            raise SystemExit("FAIL: source model changed")
        print(
            f"PASS: independently decoded scale-2 reflection, ordinary/patch output, parser bounds, "
            f"and {len(source_tensors) - 1} byte-identical non-target payloads; " +
            ("scale-1 output matches the baseline closure byte-for-byte"
             if args.baseline_quantizer else "scale-1 patch path passed"))


if __name__ == "__main__":
    main()
