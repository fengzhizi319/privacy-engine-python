// Vendored subset of github.com/fengzhizi319/PrivShield-go/pkg used by the
// TLCP terminator. Only config, tlsutil, tlsutil/certtest and crypto are
// copied; the terminator's go.mod replaces the upstream module with this
// directory so the Python repository builds without the Go monorepo.
module github.com/fengzhizi319/PrivShield-go/pkg

go 1.26.0

require (
	gitee.com/Trisia/gotlcp v1.5.0
	github.com/emmansun/gmsm v0.44.1
	github.com/youmark/pkcs8 v0.0.0-20240726163527-a2c0da244d78
	google.golang.org/grpc v1.83.2
	gopkg.in/yaml.v3 v3.0.1
)
