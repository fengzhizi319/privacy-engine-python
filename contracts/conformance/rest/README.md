# REST conformance cases

REST replay fixtures land in phase 3 together with the route/request/response alignment.
Each file follows the shared case schema in `../README.md` with `transport: "rest"` and
`request` describing `method`, `path`, `json` or `params`, plus optional `headers`.

Response comparison is a subset match: every key in `expect.json` must be present
and equal, while Python-only backward-compatible keys are allowed. `expect.status`
is checked when present. Current fixtures cover mask / mask_record / hash.
