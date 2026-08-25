#!/usr/bin/env python3
"""Exercise startup affine control vectors against a tiny GGUF server."""

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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        parsed = json.loads(raw) if raw else None
        return exc.code, parsed


def port_is_closed(port):
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def wait_ready(process, port, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness with {process.returncode}")
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
            "content",
            "tokens_predicted",
            "stopped_eos",
            "stopped_limit",
            "stopped_word",
            "stopping_word",
            "completion_probabilities",
        )
    }


def write_vector(path, layer_values, writer_type="f32"):
    from gguf import GGUFWriter

    writer = GGUFWriter(path, "controlvector")
    writer.add_string("controlvector.model_hint", "stories260K")
    writer.add_uint32("controlvector.layer_count", len(layer_values))
    for layer, values in sorted(layer_values.items()):
        if writer_type == "f16":
            payload = np.asarray(values, dtype="<f2")
        else:
            payload = np.asarray(values, dtype="<f4")
        writer.add_tensor(f"direction.{layer}", payload)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    os.chmod(path, 0o600)


def load_source_directions(path, width, layers):
    from gguf import GGUFReader

    tensors = {tensor.name: tensor for tensor in GGUFReader(path, mode="r").tensors}
    result = {}
    for layer in layers:
        name = f"direction.{layer}"
        if name not in tensors:
            raise ValueError(f"source vector lacks {name}")
        values = np.asarray(tensors[name].data, dtype="<f4").reshape(-1).copy()
        if values.shape != (width,) or not np.all(np.isfinite(values)):
            raise ValueError(f"source vector has invalid {name}")
        norm = float(np.linalg.norm(values.astype(np.float64)))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(f"source vector {name} norm is {norm}")
        result[layer] = values
    return result


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
            str(server),
            "--model", str(self.args.model),
            "--host", "127.0.0.1",
            "--port", str(self.args.port),
            "--ctx-size", "256",
            "--batch-size", "128",
            "--ubatch-size", "128",
            "--threads", "2",
            "--threads-batch", "2",
            "--parallel", "1",
            "--cache-ram", "0",
            "--alias", alias,
            *extra,
        ]

    def valid_case(self, name, server, extra, repeat=False, inspect=False):
        if not port_is_closed(self.args.port):
            raise RuntimeError(f"port {self.args.port} is already open")
        log_path = self.output / f"{name}.log"
        command = self.command(server, name, extra)
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
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
                        raise RuntimeError(f"{name} repeat returned HTTP {status}")
                    responses.append(behavioral_view(response))
                    if responses[0] != responses[1]:
                        raise RuntimeError(f"{name} graph-reuse response changed")

                state = None
                mutation_status = {}
                if inspect:
                    status, state = request_json(
                        "GET", f"{self.base_url}/control-vectors")
                    if status != 200:
                        raise RuntimeError(f"{name} control-vector GET returned {status}")
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
                    raise RuntimeError(f"{name} left port {self.args.port} open")

    def invalid_case(self, name, extra):
        if not port_is_closed(self.args.port):
            raise RuntimeError(f"port {self.args.port} is already open")
        log_path = self.output / f"invalid-{name}.log"
        command = self.command(self.args.patched_server, f"invalid-{name}", extra)
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"invalid case {name} did not fail closed") from exc
                if returncode == 0:
                    raise RuntimeError(f"invalid case {name} exited successfully")
                if not port_is_closed(self.args.port):
                    raise RuntimeError(f"invalid case {name} opened the inference port")
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
    parser.add_argument("--port", type=int, default=18091)
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

    layers = (1, 2, 3, 4)
    directions = load_source_directions(args.source_vector, 64, layers)
    projection = args.output / "projection.gguf"
    offset = args.output / "offset.gguf"
    nonunit = args.output / "nonunit.gguf"
    nonfinite = args.output / "nonfinite.gguf"
    missing_layer = args.output / "missing-layer.gguf"
    wrong_width = args.output / "wrong-width.gguf"
    wrong_type = args.output / "wrong-type.gguf"
    out_of_model = args.output / "out-of-model.gguf"

    write_vector(projection, directions)
    write_vector(offset, {layer: values * 50.0 for layer, values in directions.items()})
    write_vector(nonunit, {layer: values * 0.5 for layer, values in directions.items()})
    nonfinite_values = {layer: values.copy() for layer, values in directions.items()}
    nonfinite_values[2][0] = np.nan
    write_vector(nonfinite, nonfinite_values)
    write_vector(missing_layer, {layer: directions[layer] for layer in (1, 2, 3)})
    write_vector(wrong_width, {layer: directions[layer][:-1] for layer in layers})
    write_vector(wrong_type, directions, writer_type="f16")
    write_vector(out_of_model, {**directions, 5: directions[4]})

    matrix = Matrix(args, args.output)
    control_range = ["--control-vector-layer-range", "1", "4"]
    result = {
        "inputs": {
            str(path): sha256(path) for path in (
                args.patched_server, args.baseline_server, args.model,
                args.source_vector)
        },
        "artifacts": {
            path.name: sha256(path) for path in (
                projection, offset, nonunit, nonfinite, missing_layer,
                wrong_width, wrong_type, out_of_model)
        },
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
    valid["projection-only"] = matrix.valid_case(
        "projection-only", args.patched_server,
        ["--control-vector-projection", str(projection), *control_range],
        repeat=True, inspect=True)
    valid["offset-only"] = matrix.valid_case(
        "offset-only", args.patched_server,
        ["--control-vector", str(offset), *control_range], repeat=True)
    valid["projection-plus-offset"] = matrix.valid_case(
        "projection-plus-offset", args.patched_server,
        [
            "--control-vector-projection", str(projection),
            "--control-vector", str(offset),
            *control_range,
        ], repeat=True, inspect=True)

    def response(name):
        return valid[name]["responses"][0]

    if response("baseline-reference") != response("patched-baseline"):
        raise RuntimeError("no-vector behavior changed from the preserved baseline")
    if response("baseline-additive") != response("patched-additive"):
        raise RuntimeError("additive-only behavior changed from the preserved baseline")
    if response("projection-only") == response("patched-baseline"):
        raise RuntimeError("projection did not measurably change the response")
    if response("projection-plus-offset") in (
            response("projection-only"), response("offset-only"),
            response("patched-additive"), response("patched-baseline")):
        raise RuntimeError("projection plus offset is not behaviorally distinct")

    projection_state = valid["projection-only"]["control_vectors"]
    affine_state = valid["projection-plus-offset"]["control_vectors"]
    if ([entry.get("type") for entry in projection_state] != ["projection"]
            or [entry.get("type") for entry in affine_state]
            != ["projection", "affine_offset"]
            or not all(entry.get("read_only") for entry in affine_state)):
        raise RuntimeError("startup affine state is not detectable and read-only")

    invalid = result["invalid"]
    invalid["duplicate"] = matrix.invalid_case("duplicate", [
        "--control-vector-projection", str(projection),
        "--control-vector-projection", str(projection), *control_range])
    invalid["missing-file"] = matrix.invalid_case("missing-file", [
        "--control-vector-projection", str(args.output / "absent.gguf"), *control_range])
    invalid["nonunit"] = matrix.invalid_case("nonunit", [
        "--control-vector-projection", str(nonunit), *control_range])
    invalid["nonfinite"] = matrix.invalid_case("nonfinite", [
        "--control-vector-projection", str(nonfinite), *control_range])
    invalid["missing-layer"] = matrix.invalid_case("missing-layer", [
        "--control-vector-projection", str(missing_layer), *control_range])
    invalid["wrong-width"] = matrix.invalid_case("wrong-width", [
        "--control-vector-projection", str(wrong_width), *control_range])
    invalid["wrong-type"] = matrix.invalid_case("wrong-type", [
        "--control-vector-projection", str(wrong_type), *control_range])
    invalid["out-of-model-data"] = matrix.invalid_case("out-of-model-data", [
        "--control-vector-projection", str(out_of_model), *control_range])
    invalid["offset-missing-layer"] = matrix.invalid_case("offset-missing-layer", [
        "--control-vector-projection", str(projection),
        "--control-vector", str(missing_layer), *control_range])
    invalid["offset-nonfinite"] = matrix.invalid_case("offset-nonfinite", [
        "--control-vector-projection", str(projection),
        "--control-vector", str(nonfinite), *control_range])
    invalid["offset-out-of-model"] = matrix.invalid_case("offset-out-of-model", [
        "--control-vector-projection", str(projection),
        "--control-vector", str(out_of_model), *control_range])
    invalid["range-zero"] = matrix.invalid_case("range-zero", [
        "--control-vector-projection", str(projection),
        "--control-vector-layer-range", "0", "4"])
    invalid["range-reversed"] = matrix.invalid_case("range-reversed", [
        "--control-vector-projection", str(projection),
        "--control-vector-layer-range", "4", "3"])
    invalid["range-terminal"] = matrix.invalid_case("range-terminal", [
        "--control-vector-projection", str(projection),
        "--control-vector-layer-range", "1", "5"])

    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.chmod(result_path, 0o600)
    print("control-vector projection end-to-end tests passed")
    print(f"result={result_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"FAIL: {exc}") from exc
