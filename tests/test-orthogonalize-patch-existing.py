#!/usr/bin/env python3
"""Optional end-to-end test for copy-on-write orthogonalization patching.

This test needs an existing one-shard GGUF with a square attention-output
matrix. It is intentionally not registered with CTest because the repository
does not ship model weights.
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
from gguf import GGUFReader, GGUFWriter  # noqa: E402


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
    args = parser.parse_args()
    args.quantizer = args.quantizer.resolve()
    args.model = args.model.resolve()
    if not args.quantizer.is_file() or not os.access(args.quantizer, os.X_OK):
        parser.error("quantizer is not executable")
    if not args.model.is_file():
        parser.error("model is not readable")

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
        output_prefix = directory / "candidate"
        candidate = directory / "candidate-00001-of-00001.gguf"
        write_direction(direction, width)
        subprocess.run([
            "cp", "--reflink=always", "--preserve=mode,timestamps", "--",
            str(args.model), str(candidate),
        ], check=True)
        candidate.chmod(candidate.stat().st_mode | 0o200)
        if (args.model.stat().st_dev, args.model.stat().st_ino) == (
                candidate.stat().st_dev, candidate.stat().st_ino):
            raise SystemExit("FAIL: candidate is not a distinct inode")
        pristine_candidate_hash = sha256(candidate)

        common = [
            str(args.quantizer), "--allow-requantize", "--pure", "--keep-split",
            "--orthogonalize-control-vector", str(direction),
            "--orthogonalize-layer-range", "1", "2",
            "--orthogonalize-subspace-rank", "2",
            "--orthogonalize-patch-existing",
            "--orthogonalize-pattern", f"^{re.escape(args.target)}$",
            "--orthogonalize-scale", "1.0",
            "--orthogonalize-expected-count", "1",
            "--orthogonalize-quant-passes", "16",
            "--orthogonalize-quant-correction", "0.25",
            "--orthogonalize-max-residual", "0.10",
        ]
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

        candidate_reader, candidate_tensors = tensor_map(candidate)
        if source_reader.data_offset != candidate_reader.data_offset:
            raise SystemExit("FAIL: GGUF header length changed")
        with args.model.open("rb") as left, candidate.open("rb") as right:
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
            differs = payload_digest(source_tensor) != payload_digest(candidate_tensor)
            if differs:
                changed.append(name)
        if changed != [args.target]:
            raise SystemExit(f"FAIL: changed payload set is {changed!r}")
        print(
            f"PASS: patch-existing changed only {args.target}; complete header and "
            f"{len(source_tensors) - 1} non-target payloads remained byte-identical")


if __name__ == "__main__":
    main()
