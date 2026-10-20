// Package config provides shared environment variable helpers for console Go modules.
// Package config 为控制台各 Go 模块提供共享的环境变量读取工具函数。
//
// 三个模块（service-hub / datasource-mgr / audit-log）原先各自维护完全相同的
// getEnv / getEnvInt / getEnvBool 实现，现统一抽取至本包。
package config

import (
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// EnvString reads an environment variable, returning def if unset or empty.
// EnvString 读取环境变量，未设置或为空时返回默认值。
func EnvString(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}

// EnvStringFirstSet returns the first non-empty value among the given environment variables.
// Returns empty string if all of them are unset or empty.
// EnvStringFirstSet 依次读取给定的环境变量，返回第一个非空值；全为空时返回空字符串。
func EnvStringFirstSet(names ...string) string {
	for _, name := range names {
		if v := os.Getenv(name); v != "" {
			return v
		}
	}
	return ""
}

// EnvStringOptional reads an environment variable, distinguishing "unset" from "explicitly set to empty".
// It only falls back to def when the variable is completely unset; an empty string is treated as a valid value.
// EnvStringOptional 读取环境变量，区分"未设置"与"显式设为空字符串"；
// 仅在变量完全未设置时才使用默认值，空字符串被视为合法值。
func EnvStringOptional(name, def string) string {
	if v, ok := os.LookupEnv(name); ok {
		return v
	}
	return def
}

// EnvInt reads an environment variable as int, returning def on missing or invalid.
// EnvInt 以整数形式读取环境变量，缺失或无效时返回默认值。
func EnvInt(name string, def int) int {
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	i, err := strconv.Atoi(v)
	if err != nil {
		warnMalformed(name, v, def)
		return def
	}
	return i
}

// EnvInt64 reads an environment variable as int64, returning def on missing or invalid.
// EnvInt64 以 64 位整数形式读取环境变量，缺失或无效时返回默认值。
func EnvInt64(name string, def int64) int64 {
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	i, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		warnMalformed(name, v, def)
		return def
	}
	return i
}

// EnvFloat reads an environment variable as float64, returning def on missing or invalid.
// EnvFloat 以浮点数形式读取环境变量，缺失或无效时返回默认值。
//
// strconv.ParseFloat 对 "NaN"/"Inf"/"+Inf" **不报错**，但它们一旦落到安全阈值上
// 就是静默漂移：NaN 与任何数值的比较均为 false，用它做速率下限校验
// （`rps <= 0` 放行）会让限流整条失效（实测 PRIVACY_RATE_LIMIT_DEFAULT_RPS=NaN
// → 全部请求放行），转成整数则是 -9223372036854775808。非有限值一律按
// 「取值非法」处理：告警并回退默认值。
func EnvFloat(name string, def float64) float64 {
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	f, ok := finiteFloat(v)
	if !ok {
		warnMalformed(name, v, def)
		return def
	}
	return f
}

// finiteFloat 解析浮点读数，ok=false 表示取值非法或为非有限值（NaN/±Inf）。
func finiteFloat(v string) (float64, bool) {
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, false
	}
	return f, true
}

// EnvBool reads an environment variable as bool.
// Recognized true values: "true", "1", "yes", "on" (case-insensitive).
// Recognized false values: "false", "0", "no", "off" (case-insensitive).
// EnvBool 以布尔值读取环境变量。
// 识别为 true 的值: "true"、"1"、"yes"、"on"（不区分大小写）。
// 识别为 false 的值: "false"、"0"、"no"、"off"（不区分大小写）。
//
// 取值既不是 true 词表也不是 false 词表时（`ture` 拼错、`2`、`enable`、尾随空格
// 之外的任何形态）**返回默认值并告警**，而不是静默判 false：
// 本包的布尔读数几乎全是安全开关（*_AUTH_ENABLED / *_SELF_REGISTER /
// *_ADMIN_STEPUP_REQUIRED / *_STRICT），默认值多为 true。
// 旧实现在这里 return false，等于「把开关写错字就等于把它关掉」——
// 与同仓库 *_API_KEY_ROTATION_GRACE「取值非法保持默认并 WARN，不静默放宽」
// 的既定口径相悖，是 fail-closed 体系里唯一一处 fail-open 的读数。
func EnvBool(name string, def bool) bool {
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	b, ok := boolValue(v)
	if !ok {
		warnMalformedBool(name, v, def)
		return def
	}
	return b
}

// EnvStringFallback reads primary env var; if unset or empty, falls back to fallback env var; if still unset, returns def.
// EnvStringFallback 优先读取主环境变量；若未设置或为空则回退读取次环境变量；仍未设置则返回默认值。
func EnvStringFallback(primary, fallback, def string) string {
	if v := os.Getenv(primary); v != "" {
		return v
	}
	if v := os.Getenv(fallback); v != "" {
		return v
	}
	return def
}

// EnvIntFallback reads primary env var as int; if unset or invalid, falls back to fallback env var; if still unset, returns def.
// EnvIntFallback 优先读取主环境变量为整数；若未设置或无效则回退读取次环境变量；仍未设置则返回默认值。
func EnvIntFallback(primary, fallback string, def int) int {
	if v := os.Getenv(primary); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	if v := os.Getenv(fallback); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return def
}

// EnvBoolFallback reads primary env var as bool; if unset, empty or **unrecognized**,
// falls back to fallback env var; if still unset or unrecognized, returns def.
// EnvBoolFallback 优先读取主环境变量为布尔值；未设置、为空或**取值无法识别**时回退读取
// 次环境变量；仍取不到合法值时返回默认值。
//
// 「主变量写了个非法值就直接 return false」是旧实现的第二个 fail-open：
// 同函数族的 EnvIntFallback / EnvFloatFallback 遇到非法值都会继续往下找，
// 布尔读数没有理由例外 —— 这里回退到的往往才是编排里真正生效的那份配置。
func EnvBoolFallback(primary, fallback string, def bool) bool {
	if v := os.Getenv(primary); v != "" {
		if b, ok := boolValue(v); ok {
			return b
		}
		warnMalformedBool(primary, v, def)
	}
	if v := os.Getenv(fallback); v != "" {
		if b, ok := boolValue(v); ok {
			return b
		}
		warnMalformedBool(fallback, v, def)
	}
	return def
}

// EnvFloatFallback reads primary env var as float64; if unset or invalid, falls back to fallback env var; if still unset, returns def.
// 非有限值（NaN/±Inf）与解析失败同等对待：不采纳该读数，继续走回退链。
func EnvFloatFallback(primary, fallback string, def float64) float64 {
	if v := os.Getenv(primary); v != "" {
		if f, ok := finiteFloat(v); ok {
			return f
		}
	}
	if v := os.Getenv(fallback); v != "" {
		if f, ok := finiteFloat(v); ok {
			return f
		}
	}
	return def
}

// boolValue 解析布尔读数，ok=false 表示取值不在真/假词表内（而非「假」）。
func boolValue(v string) (bool, bool) {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "true", "1", "yes", "on":
		return true, true
	case "false", "0", "no", "off":
		return false, true
	default:
		return false, false
	}
}

// warnMalformedBool 把「取值畸形」变成启动日志里可见的一条告警，而不是一个静默的 false。
func warnMalformedBool(name, v string, def bool) {
	slog.Warn("malformed boolean environment value; keeping default",
		"variable", name, "value", v, "default", def)
}

// warnMalformed 把数值读数的「取值畸形」与「未设置」区分开：两者都回落默认值，
// 但只有前者说明运维写了东西、只是写错了 —— 没有这条告警，端口配成 "807x"
// 与端口根本没配在现象上完全一样。
func warnMalformed(name, v string, def any) {
	slog.Warn("malformed environment value; keeping default",
		"variable", name, "value", v, "default", def)
}

// EnvStringSlice reads a comma-separated environment variable as []string.
// Returns nil if the variable is empty.
// EnvStringSlice 以逗号分隔读取环境变量为字符串切片。
// 环境变量为空时返回 nil。
func EnvStringSlice(name string) []string {
	v := os.Getenv(name)
	if strings.TrimSpace(v) == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// RepoRelativeDefault 把「相对仓库根」的默认路径解析成可用路径。
//
// ==============================================================================
// 【为什么需要它】
// 国密 TLCP 默认开启且缺证书即拒启，故默认证书路径必须真的能解析到 ——
// 否则裸 `go run` 与所有用默认配置的测试都起不来。
//
// 但「相对仓库根」的路径只在 cwd 恰好是仓库根时成立：
//   - `go run ./services/audit-log/cmd/server`（cwd=仓库根）→ 成立
//   - `go test ./services/audit-log/internal/config/`（cwd=包目录）→ 不成立
//
// 故从 cwd 逐级上溯找 go.work，找到就据此拼绝对路径。
// 找不到（如容器内只有二进制）时原样返回 —— 那种环境本就应由
// 环境变量显式给出绝对路径（/certs/tlcp/...），不依赖本函数。
//
// 只用于**默认值**。显式配置的路径不经过这里，配错了就该直接失败。
func RepoRelativeDefault(rel string) string {
	if rel == "" || filepath.IsAbs(rel) {
		return rel
	}
	root, ok := repoCheckoutRoot()
	if !ok {
		return rel
	}
	return filepath.Join(root, rel)
}

// repoCheckoutRoot 沿 cwd 上溯找 go.work，返回仓库检出根目录。
// 与 RepoRelativeDefault 共用同一判据：一处定义"什么叫仓内"，两处才不会各说各话。
func repoCheckoutRoot() (string, bool) {
	dir, err := os.Getwd()
	if err != nil {
		return "", false
	}
	for i := 0; i < 10; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.work")); err == nil {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", false
}

// IsRepoCheckoutPath 报告 p 是否落在**本机的开发工作树**内。
//
// 判据与 RepoRelativeDefault 完全同源（同一 go.work 上溯），因此它精确回答的是
// "这个路径是不是那套缺省测试材料"：
//   - `RepoRelativeDefault` 在开发机上把 `certs/tlcp/*` 拼成的绝对路径 → true；
//   - 容器里的 `/certs/...`、`/etc/privshield/...`、命名卷上的 `/app/data/...` → false；
//   - 相对路径按 cwd 归一后再比（容器 cwd 不在检出内，故相对路径也不会误判成仓内）。
//
// 上溯不到 go.work（纯二进制容器）时一律 false：那种环境本来就不可能有仓内材料，
// 把"判不出"当成"命中"会拒掉正常部署。
func IsRepoCheckoutPath(p string) bool {
	p = strings.TrimSpace(p)
	if p == "" {
		return false
	}
	root, ok := repoCheckoutRoot()
	if !ok {
		return false
	}
	abs := p
	if !filepath.IsAbs(abs) {
		cwd, err := os.Getwd()
		if err != nil {
			return false
		}
		abs = filepath.Join(cwd, abs)
	}
	abs = filepath.Clean(abs)
	return abs == root || strings.HasPrefix(abs, root+string(filepath.Separator))
}

// ValidateMaterialOutsideRepoCheckout 断言一批凭据/TLS 材料路径**不在**开发工作树内。
//
// 为什么必须是启动门禁：各服务的 TLS 材料缺省经 RepoRelativeDefault 指向仓内
// `certs/tlcp/*`（throwaway 测试材料，随镜像与源码一起进 git）。生产部署若漏注入
// 路径变量，进程会**静默用测试证书起服务**并自称就绪 —— 材料本身可换，但"生产在用仓内
// 测试材料"这件事必须由启动期说话，而不是等审计发现集群里跑着一把公开在 git 里的私钥。
//
// 判定用**窄**口径 IsProductionLabeled（G-16：窄=启动门禁、宽=运行态），且检查所有服务的
// 标注变量（同一部署里它们描述的是同一件事，任一标生产即为生产）。非生产标注直接返回 nil：
// 开发机与 `go test` 的全部缺省都靠这套仓内材料，加一道无差别的断言等于把开发态打死。
//
// materials 的 key 必须是**环境变量名**（不是结构体字段名）：报错要能指向运维该改的那一行。
func ValidateMaterialOutsideRepoCheckout(serviceName string, materials map[string]string) error {
	if !IsProductionLabeled("APP_ENV", "PRIVACY_ENV", "PRIVACY_GATEWAY_ENV", "SERVICE_HUB_ENV", "AUDIT_LOG_ENV", "DATASOURCE_MGR_ENV") {
		return nil
	}
	var offenders []string
	for _, name := range sortedMaterialKeys(materials) {
		if IsRepoCheckoutPath(materials[name]) {
			offenders = append(offenders, fmt.Sprintf("%s=%s", name, materials[name]))
		}
	}
	if len(offenders) == 0 {
		return nil
	}
	return fmt.Errorf("%s: %d credential/TLS material path(s) point inside the development repository checkout "+
		"(throwaway test assets committed to git) while running under a production label: %s — "+
		"set these variables to the mounted production paths instead of relying on the repo-relative defaults",
		serviceName, len(offenders), strings.Join(offenders, ", "))
}

// sortedMaterialKeys 返回 materials 中**值非空**的 key（排序保证报错文案稳定、可逐字比对）。
// 空值表示"该材料未配置"，由各自的必填校验处置，不属于"用了仓内材料"，
// 也不该在这里产生一份"看起来检查过"的假覆盖。
func sortedMaterialKeys(materials map[string]string) []string {
	keys := make([]string, 0, len(materials))
	for k, v := range materials {
		if strings.TrimSpace(v) != "" {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	return keys
}
