#!/usr/bin/env python3
"""Exercise immutable affine-subspace startup state against a tiny server."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


METHOD = "k3-v9-q5-rank7-affine-v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(values):
    return hashlib.sha256(
        np.asarray(values, dtype="<f4").tobytes(order="C")).hexdigest()


def request_json(method, url, body=None, timeout=5):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, json.loads(raw) if raw else None


def port_is_closed(port):
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def wait_ready(process, port, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited before readiness with {process.returncode}")
        try:
            status, body = request_json(
                "GET", f"http://127.0.0.1:{port}/health", timeout=1)
            if status == 200 and body.get("status") == "ok":
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise RuntimeError("server readiness timed out")


def stop_process(process):
    if process.poll() is not None:
        return process.returncode
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    return process.returncode


def behavioral_view(response):
    return {
        key: response.get(key)
        for key in (
            "content", "tokens_predicted", "stopped_eos", "stopped_limit",
            "stopped_word", "stopping_word", "completion_probabilities")
    }


def write_affine(
        path,
        basis,
        offset,
        *,
        layer=2,
        architecture="controlvectorsubspace",
        model_hint="kimi-k3",
        method=METHOD,
        rank=None,
        alpha=0.0,
        alpha_type="float32",
        capture_hash="a" * 64,
        basis_hash="b" * 64,
        offset_hash=None,
        tensor_count=2,
        basis_type="f32",
        offset_type="f32",
        omit=(),
        include_basis=True,
        include_offset=True,
        extra_tensor=False):
    from gguf import GGUFWriter

    basis = np.asarray(basis)
    offset = np.asarray(offset)
    rank = basis.shape[0] if rank is None else rank
    offset_hash = payload_sha256(offset) if offset_hash is None else offset_hash
    writer = GGUFWriter(path, architecture)
    metadata = {
        "model_hint": ("string", model_hint),
        "method": ("string", method),
        "layer": ("uint32", layer),
        "rank": ("uint32", rank),
        "alpha": (alpha_type, alpha if alpha_type == "float32" else str(alpha)),
        "source_activation_sha256": ("string", capture_hash),
        "source_basis_sha256": ("string", basis_hash),
        "offset_payload_sha256": ("string", offset_hash),
        "tensor_count": ("uint32", tensor_count),
    }
    for suffix, (kind, value) in metadata.items():
        if suffix in omit:
            continue
        key = f"controlvectorsubspace.{suffix}"
        getattr(writer, f"add_{kind}")(key, value)
    if include_basis:
        dtype = "<f2" if basis_type == "f16" else "<f4"
        writer.add_tensor(f"basis.{layer}", np.asarray(basis, dtype=dtype))
    if include_offset:
        dtype = "<f2" if offset_type == "f16" else "<f4"
        writer.add_tensor(f"offset.{layer}", np.asarray(offset, dtype=dtype))
    if extra_tensor:
        writer.add_tensor("unexpected.1", np.zeros(1, dtype="<f4"))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    os.chmod(path, 0o600)


class Matrix:
    def __init__(self, args, output):
        self.args = args
        self.output = output
        self.base_url = f"http://127.0.0.1:{args.port}"
        self.request = {
            "prompt": "Once upon a time",
            "n_predict": 32,
            "temperature": -1,
            "seed": 42,
            "n_probs": 5,
            "cache_prompt": False,
            "stream": False,
        }

    def command(self, server, alias, extra):
        return [
            str(server), "--model", str(self.args.model),
            "--host", "127.0.0.1", "--port", str(self.args.port),
            "--ctx-size", "256", "--batch-size", "128",
            "--ubatch-size", "128", "--threads", "2",
            "--threads-batch", "2", "--parallel", "1", "--cache-ram", "0",
            "--alias", alias, *extra,
        ]

    def valid_case(self, name, server, extra, repeat=False, inspect=False):
        if not port_is_closed(self.args.port):
            raise RuntimeError(f"port {self.args.port} is already open")
        command = self.command(server, name, extra)
        with (self.output / f"{name}.log").open("xb") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_ready(process, self.args.port)
                status, response = request_json(
                    "POST", f"{self.base_url}/completion", self.request, timeout=30)
                if status != 200:
                    raise RuntimeError(f"{name} completion returned HTTP {status}")
                responses = [behavioral_view(response)]
                if repeat:
                    status, response = request_json(
                        "POST", f"{self.base_url}/completion", self.request, timeout=30)
                    if status != 200:
                        raise RuntimeError(
                            f"{name} repeated completion returned HTTP {status}")
                    responses.append(behavioral_view(response))
                    if responses[0] != responses[1]:
                        raise RuntimeError(f"{name} graph-reuse response changed")
                state = None
                mutation_status = {}
                if inspect:
                    status, state = request_json(
                        "GET", f"{self.base_url}/control-vectors")
                    if status != 200:
                        raise RuntimeError(f"{name} state returned HTTP {status}")
                    for endpoint in ("load", "unload", "apply"):
                        status, _ = request_json(
                            "POST", f"{self.base_url}/control-vectors/{endpoint}", {})
                        mutation_status[endpoint] = status
                        if status != 409:
                            raise RuntimeError(
                                f"{name} hot {endpoint} returned {status}, expected 409")
                return {
                    "command": command,
                    "responses": responses,
                    "control_vectors": state,
                    "hot_mutation_status": mutation_status,
                }
            finally:
                returncode = stop_process(process)
                if returncode != 0:
                    raise RuntimeError(f"{name} server exited with {returncode}")
                if not port_is_closed(self.args.port):
                    raise RuntimeError(f"{name} left port open")

    def invalid_case(self, name, extra):
        if not port_is_closed(self.args.port):
            raise RuntimeError(f"port {self.args.port} is already open")
        command = self.command(self.args.patched_server, f"invalid-{name}", extra)
        with (self.output / f"invalid-{name}.log").open("xb") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT)
            try:
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"invalid case {name} did not fail closed") from exc
                if returncode == 0:
                    raise RuntimeError(f"invalid case {name} exited successfully")
                if not port_is_closed(self.args.port):
                    raise RuntimeError(f"invalid case {name} opened the port")
                return {"command": command, "returncode": returncode}
            finally:
                stop_process(process)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patched-server", type=Path, required=True)
    parser.add_argument("--baseline-server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-vector", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18092)
    args = parser.parse_args()
    os.umask(0o077)

    for path in (
            args.patched_server, args.baseline_server, args.model,
            args.source_vector):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not os.access(args.patched_server, os.X_OK):
        raise SystemExit(f"patched server is not executable: {args.patched_server}")
    if not os.access(args.baseline_server, os.X_OK):
        raise SystemExit(f"baseline server is not executable: {args.baseline_server}")
    if not (args.gguf_py / "gguf").is_dir():
        raise SystemExit(f"invalid gguf-py path: {args.gguf_py}")
    if args.output.exists():
        raise SystemExit(f"refusing to reuse output: {args.output}")
    args.output.mkdir(mode=0o700)
    sys.path.insert(0, str(args.gguf_py))

    width = 64
    layer = 2
    rank = 7
    basis = np.zeros((rank, width), dtype="<f4")
    for row in range(rank):
        basis[row, row] = 1.0
    offset = np.zeros(width, dtype="<f4")
    offset[0] = 50.0
    valid_path = args.output / "affine.gguf"
    write_affine(valid_path, basis, offset)

    malformed = {}

    def artifact(name, **kwargs):
        path = args.output / f"{name}.gguf"
        artifact_basis = kwargs.pop("basis", basis)
        artifact_offset = kwargs.pop("offset", offset)
        write_affine(path, artifact_basis, artifact_offset, **kwargs)
        malformed[name] = path

    artifact("wrong-architecture", architecture="controlvector")
    artifact("wrong-model", model_hint="other")
    artifact("wrong-method", method="other")
    artifact("missing-method", omit=("method",))
    artifact("layer-zero", layer=0)
    artifact("layer-terminal", layer=5)
    artifact("rank-zero", rank=0)
    artifact("rank-too-large", rank=65)
    artifact("rank-shape-mismatch", rank=3)
    artifact("alpha-wrong-type", alpha_type="string")
    artifact("tensor-count-mismatch", tensor_count=3)
    artifact("wrong-width", basis=basis[:, :-1], offset=offset[:-1])
    artifact("basis-f16", basis_type="f16")
    artifact("offset-f16", offset_type="f16")
    artifact("missing-basis", include_basis=False, tensor_count=1)
    artifact("missing-offset", include_offset=False, tensor_count=1)
    artifact("extra-tensor", extra_tensor=True, tensor_count=3)
    nonfinite = basis.copy()
    nonfinite[0, 0] = np.nan
    artifact("nonfinite-basis", basis=nonfinite)
    nonunit = basis.copy()
    nonunit[0] *= 0.5
    artifact("nonunit-basis", basis=nonunit)
    nonorthogonal = basis.copy()
    nonorthogonal[1] = nonorthogonal[0]
    artifact("nonorthogonal-basis", basis=nonorthogonal)
    nonfinite_offset = offset.copy()
    nonfinite_offset[0] = np.inf
    artifact("nonfinite-offset", offset=nonfinite_offset)
    outside = offset.copy()
    outside[rank] = 1.0
    artifact("offset-outside-span", offset=outside)
    artifact("offset-hash", offset_hash="0" * 64)
    artifact("source-hash", capture_hash="INVALID")
    artifact("source-hash-uppercase", basis_hash="B" * 64)
    artifact("alpha-nonfinite", alpha=np.nan)

    matrix = Matrix(args, args.output)
    affine_args = ["--control-vector-affine-subspace", str(valid_path)]
    control_range = ["--control-vector-layer-range", "1", "4"]
    result = {
        "inputs": {str(path): sha256(path) for path in (
            args.patched_server, args.baseline_server, args.model,
            args.source_vector)},
        "artifacts": {
            path.name: sha256(path)
            for path in [valid_path, *malformed.values()]},
        "valid": {},
        "invalid": {},
    }
    valid = result["valid"]
    valid["baseline-reference"] = matrix.valid_case(
        "baseline-reference", args.baseline_server, [])
    valid["patched-baseline"] = matrix.valid_case(
        "patched-baseline", args.patched_server, [], repeat=True)
    valid["baseline-additive"] = matrix.valid_case(
        "baseline-additive", args.baseline_server,
        ["--control-vector-scaled", str(args.source_vector), "50", *control_range])
    valid["patched-additive"] = matrix.valid_case(
        "patched-additive", args.patched_server,
        ["--control-vector-scaled", str(args.source_vector), "50", *control_range],
        repeat=True)
    valid["affine-subspace"] = matrix.valid_case(
        "affine-subspace", args.patched_server, affine_args,
        repeat=True, inspect=True)

    def response(name):
        return valid[name]["responses"][0]

    if response("baseline-reference") != response("patched-baseline"):
        raise RuntimeError("no-vector behavior changed from baseline")
    if response("baseline-additive") != response("patched-additive"):
        raise RuntimeError("additive-only behavior changed from baseline")
    if response("affine-subspace") in (
            response("patched-baseline"), response("patched-additive")):
        raise RuntimeError("affine-subspace behavior is not distinct")
    state = valid["affine-subspace"]["control_vectors"]
    if (not isinstance(state, list) or len(state) != 1
            or state[0].get("path") != str(valid_path)
            or state[0].get("type") != "affine_subspace"
            or state[0].get("layer") != layer
            or state[0].get("rank") != rank
            or state[0].get("alpha") != 0.0
            or state[0].get("read_only") is not True
            or state[0].get("applied") is not True):
        raise RuntimeError(f"affine startup state is incomplete: {state}")

    invalid = result["invalid"]
    invalid["missing-file"] = matrix.invalid_case(
        "missing-file", ["--control-vector-affine-subspace",
                         str(args.output / "absent.gguf")])
    invalid["duplicate"] = matrix.invalid_case(
        "duplicate", [*affine_args, *affine_args])
    invalid["with-additive"] = matrix.invalid_case(
        "with-additive", [*affine_args, "--control-vector", str(args.source_vector)])
    invalid["with-scaled"] = matrix.invalid_case(
        "with-scaled", [*affine_args, "--control-vector-scaled",
                        str(args.source_vector), "1"])
    invalid["with-projection"] = matrix.invalid_case(
        "with-projection", [*affine_args, "--control-vector-projection",
                            str(args.source_vector)])
    invalid["with-range"] = matrix.invalid_case(
        "with-range", [*affine_args, *control_range])
    for name, path in malformed.items():
        invalid[name] = matrix.invalid_case(
            name, ["--control-vector-affine-subspace", str(path)])

    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.chmod(result_path, 0o600)
    print("control-vector affine-subspace end-to-end tests passed")
    print(f"result={result_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"FAIL: {exc}") from exc
