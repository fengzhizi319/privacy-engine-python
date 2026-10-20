module github.com/fengzhizi319/PrivShield-go/services/privacy-engine-python/terminator

go 1.26.0

require (
	github.com/emmansun/gmsm v0.44.1
	github.com/fengzhizi319/PrivShield-go/pkg v0.0.0
	google.golang.org/grpc v1.83.2
)

require (
	gitee.com/Trisia/gotlcp v1.5.0 // indirect
	github.com/youmark/pkcs8 v0.0.0-20240726163527-a2c0da244d78 // indirect
	golang.org/x/crypto v0.56.0 // indirect
	golang.org/x/net v0.58.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.41.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260526163538-3dc84a4a5aaa // indirect
	google.golang.org/protobuf v1.36.11 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/fengzhizi319/PrivShield-go/pkg => ./third_party/pkg
