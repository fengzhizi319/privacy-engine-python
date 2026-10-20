# gRPC conformance cases

gRPC replay fixtures land in phase 4 together with the RPC/status alignment.
Each file follows the shared case schema in `../README.md` with `transport: "grpc"` and
`request` describing `rpc` plus the protobuf request message in JSON form.

Responses are marshalled with protojson (Go) / `MessageToDict` (Python) and
compared with the same subset rule as REST. Current fixtures cover Mask /
MaskRecord / Hash.
