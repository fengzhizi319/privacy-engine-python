# Cross-language conformance fixtures

These fixtures are the shared contract between the Go reference engine
(`services/privacy-engine`) and the Python engine (`services/privacy-engine-python`).
Both sides load the same JSON files and must agree.

## Layout

```text
contracts/conformance/
  manifest.yaml       # suite registry and status
  primitives/*.json   # one case per file
  rest/*.json         # REST replay cases (phase 3)
  grpc/*.json         # gRPC replay cases (phase 4)
```

## Case schema

```json
{
  "id": "mask.phone.cn",
  "kind": "exact",
  "transport": "primitive",
  "request": {
    "op": "mask_phone",
    "args": {"value": "13812345678"}
  },
  "expect": {"result": "138****5678"},
  "tolerance": 0
}
```

- `kind`:
  - `exact` — deterministic scalar/object result, compared exactly (floats use `tolerance`).
  - `exact_unordered` — deterministic result whose records may be emitted in any order;
    `expect.records` is compared as a multiset.
  - `invariant` — only `expect.invariants` are asserted. Required for DP/LDP/QOL because
    Go uses `crypto/rand` / unseeded `math/rand/v2` and cannot be reproduced sample-by-sample.
  - `error` — both sides must reject the request; `expect.error_contains` is matched against
    the error message (Go) or exception message (Python).
- `transport`: `primitive` | `rest` | `grpc`.
- `tolerance`: absolute/relative tolerance for float comparison (0 = exact).

## Rules

1. Go is the source of truth for shared behavior. A fixture must be generated from Go
   behavior, never hand-invented to match Python.
2. Never freeze random samples. DP/LDP/QOL fixtures assert mechanism, bounds, counts and
   distribution invariants only.
3. Add a fixture with every shared-contract change; do not weaken an existing fixture to
   make an implementation pass.
4. Python-only superset behavior (MLX, Arrow IPC, Redis/SQLite budget, extra pipelines)
   does not belong here unless the Go engine also exposes the same contract.
