// Package tlsutil — dynamic mTLS CN whitelist with hot-reload support.
// Package tlsutil — 动态 mTLS 客户端 CN（Common Name）白名单与热重载管理组件。
//
// ==============================================================================
// 【核心能力与架构设计】
// 1. 【细粒度 Scope 访问控制 (Per-CN Method Scope)】：
//    从 YAML 配置文件加载客户端证书 CN 及其允许调用的 RPC 方法集合（如 `["/PrivacyService/Process"]`
//    或通配符 `["*"]` / `["/AuditLog/*"]`）；
// 2. 【5 秒无依赖热重载 (Zero-Dependency Hot-Reload)】：
//    后台独立协程每 5 秒通过 os.Stat 轮询配置文件的修改时间（ModTime），
//    在检测到文件被编辑或 ConfigMap 挂载更新时自动触发安全重载，无需重启进程或引入 fsnotify 外部依赖；
// 3. 【读写锁并发保护 (RWMutex Concurrency)】：
//    在鉴权热路径（CheckScope / IsAuthorized）上使用读锁（RLock）保证微秒级高性能并发查询，
//    在配置重载（reload）时使用写锁（Lock）实现原子全量替换，消除读写数据竞争；
// 4. 【单一配置格式】：仅接受 `clients: [{cn, allowed_scopes}]`。
//
// ==============================================================================
// 【白名单配置文件格式范例 (config/mtls-whitelist.yaml)】
//
//	version: "1.0"
//	clients:
//	  - cn: "service-hub.privshield.internal"
//	    allowed_scopes:
//	      - "/PrivacyService/Process"
//	      - "/AuditLog/*"
//	    role: "orchestrator"
//	    description: "数据服务调度中枢核心客户端"
//	    enabled: true
//	  - cn: "bff-go.privshield.internal"
//	    allowed_scopes: ["*"]
//	    role: "gateway"
//	    enabled: true
// ==============================================================================

package tlsutil

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
)

// WhitelistClient represents a single CN entry in the whitelist config.
// WhitelistClient 表示白名单配置文件中的单个客户端 CN 条目。
type WhitelistClient struct {
	// CN 为客户端证书的 Subject Common Name（必填）。
	CN string `yaml:"cn"`

	// AllowedScopes 为该客户端允许调用的 gRPC 全路径方法列表（如 ["/PrivacyService/Process", "/AuditLog/*"]）。
	AllowedScopes []string `yaml:"allowed_scopes"`

	// CertFingerprints 为该 CN 允许的客户端证书 SHA-256 指纹列表（hex，可带冒号分隔）。
	// 非空时执行证书指纹钉扎：仅当对端叶证书指纹命中本列表才放行，
	// 防止同一 CA 下冒用 CN 的「同名证书」绕过白名单（零信任通道准入加固）。
	// 留空表示不启用钉扎（保持既有 CN 匹配语义）。
	CertFingerprints []string `yaml:"cert_fingerprints,omitempty"`

	// Role 为客户端的角色标识（可选，如 "orchestrator", "gateway"）。
	Role string `yaml:"role,omitempty"`

	// Description 为条目的可读性描述信息（可选）。
	Description string `yaml:"description,omitempty"`

	// Enabled 表示是否启用该条目（nil 或 true 表示启用，false 表示临时禁用）。
	Enabled *bool `yaml:"enabled,omitempty"`
}

// WhitelistConfig represents the top-level whitelist YAML configuration.
// WhitelistConfig 表示白名单 YAML 配置文件的顶层根结构。
//
// Only clients with allowed_scopes are accepted.
type WhitelistConfig struct {
	Version       string            `yaml:"version"`
	Clients       []WhitelistClient `yaml:"clients"`
	DefaultScopes []string          `yaml:"default_scopes"`
}

// DynamicWhitelist manages a hot-reloadable CN → scopes mapping.
// DynamicWhitelist 管理线程安全、可热重载的客户端 CN 到 AllowedScopes 的映射字典。
type DynamicWhitelist struct {
	mu            sync.RWMutex
	clients       map[string][]string // CN → allowed scopes 切片
	certFPs       map[string][]string // CN → 证书 SHA-256 指纹钉扎列表（空 = 不启用钉扎）
	defaultScopes []string            // 默认 scope
	path          string              // 配置文件物理路径
	fingerprint   whitelistFingerprint

	// 轮询与停机状态
	stopCh  chan struct{}
	stopped bool
	stopMu  sync.Mutex

	// 内容 hash 抽检节流（unix 纳秒）：CheckReload 热路径上 mtime/size 未变时，
	// 至多每 2 秒读一次文件计算 sha256，防「保留 mtime 的等尺寸篡改」。
	lastHashCheck atomic.Int64
}

// whitelistFingerprint 记录白名单文件的 mtime + size + 内容 sha256 三元指纹。
type whitelistFingerprint struct {
	modTime time.Time
	size    int64
	hash    string
}

// whitelistHashThrottle 是内容 hash 抽检的最小间隔。
const whitelistHashThrottle = 2 * time.Second

// NewDynamicWhitelist creates a whitelist from a YAML file and starts background polling.
//
// NewDynamicWhitelist 从指定路径加载 YAML 白名单文件并拉起后台 5 秒轮询重载协程：
// 1. 立即同步执行首次 reload 解析配置文件；
// 2. 若解析成功，在后台启动 poll 协程持续监听文件 mtime 变更；
// 3. 服务停机时应调用 Close() 优雅释放轮询协程。
func NewDynamicWhitelist(path string) (*DynamicWhitelist, error) {
	cleanPath := filepath.Clean(path)
	dw := &DynamicWhitelist{
		clients: make(map[string][]string),
		certFPs: make(map[string][]string),
		path:    cleanPath,
		stopCh:  make(chan struct{}),
	}
	if err := dw.reload(); err != nil {
		return nil, err
	}
	go dw.poll()
	return dw, nil
}

// NewStaticWhitelist creates an in-memory whitelist from a static slice of CNs with wildcard scopes.
func NewStaticWhitelist(cns []string) *DynamicWhitelist {
	clients := make(map[string][]string, len(cns))
	for _, cn := range cns {
		if cn != "" {
			clients[cn] = []string{"*"}
		}
	}
	return &DynamicWhitelist{
		clients: clients,
		certFPs: make(map[string][]string),
	}
}

// mntReadOnlyFlag 是 Linux uapi `linux/mount.h` 的 MNT_RDONLY 位（Go 的 syscall 包未导出该常量）。
const mntReadOnlyFlag = 0x00000001

// volumeReadOnly 实测 path 所在卷的挂载标志，返回它是否以只读挂载。
//
// 判据来自 statfs 而不是编排声明：Pod spec 里写 `readOnly: true` 不等于该路径落在这块卷上
// （ConfigMap 投影、subPath、镜像层里的同名文件都会让声明与事实分开）。探不到就报 false，
// 让调用方按"无法证明风险已消除"处置。
func volumeReadOnly(path string) bool {
	var st syscall.Statfs_t
	if err := syscall.Statfs(filepath.Dir(path), &st); err != nil {
		return false
	}
	return st.Flags&mntReadOnlyFlag != 0
}

// shouldTolerateChmodFailure 判定一次白名单文件 chmod 失败可否容忍（不阻断加载）。
// 两个条件**同时**成立才容忍，缺一即照旧拒启：
//   - readOnlyMount：文件所在卷确实只读（由 volumeReadOnly 实测）；
//   - err 是 EROFS：失败的原因就是卷只读。
//
// 双条件不是冗余：只读卷 + 其他错误（EPERM/EACCES/ENOMEM…）说明失败另有原因，
// 一并放行会把真正的加固失效伪装成"只读所以没事"；EROFS 但 statfs 判不出只读时也不放行，
// 宁可拒启也不在"无法证明风险已消除"时沉默。
//
// 容忍的是**写位风险**：只读挂载让同机其他账号无法改写这份授权输入，篡改路径被结构性消除。
// **读位风险仍在**——0644 意味着本机任意账号可读走 CN 与 scope 清单。因此这里必须留痕 WARN，
// 且修复方向永远是"源资产就设成 0600"，不是"给挂载补个 :ro"。
func shouldTolerateChmodFailure(err error, readOnlyMount bool) bool {
	return readOnlyMount && errors.Is(err, syscall.EROFS)
}

// reload reads and parses the YAML whitelist configuration file.
//
// reload 在写锁保护下读取并解析 YAML 白名单，并执行三道文件加固校验：
//  1. Lstat 拒绝符号链接（防通过软链把白名单偷换成外部文件）；
//  2. 文件 mode 放开组/其他读写位时 WARN 并收敛为 0600；只读挂载上降为 WARN 放行
//     （见 shouldTolerateChmodFailure），不再因此拒启；
//  3. 解析结果为空而现有配置非空时拒绝替换（保旧配置）并打 ERROR；
//  4. 成功路径记录 mtime + size + 内容 sha256 三元指纹供增量检测。
//     第 2 道在只读挂载上降为 WARN 放行（见 shouldTolerateChmodFailure），不再拒启。
func (dw *DynamicWhitelist) reload() error {
	// 符号链接拒绝：Lstat 不跟随软链， ModeSymlink 直接报错
	lstatInfo, err := os.Lstat(dw.path)
	if err != nil {
		return err
	}
	if lstatInfo.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("refusing to load whitelist via symbolic link: %s", dw.path)
	}

	// 权限收敛：组/其他任一读写执行位被放开即告警并收敛 0600
	if perm := lstatInfo.Mode().Perm(); perm&0o077 != 0 {
		log.Printf("[mTLS Whitelist] WARN: whitelist file %s has permissive mode %o, tightening to 0600", dw.path, perm)
		if chmodErr := os.Chmod(dw.path, 0o600); chmodErr != nil {
			// 只读挂载上的收敛失败可容忍，但必须说话（G7「降档必须说话」）。
			// Helm 把 whitelist 以 `:ro` 挂进来时 chmod 必然 EROFS，原实现直接拒启，
			// 于是"资产文件模式太宽"这一条告警被放大成 Pod 崩溃循环
			// （ISSUE-018 形态②在 Helm 侧的残留形态）。
			if shouldTolerateChmodFailure(chmodErr, volumeReadOnly(dw.path)) {
				log.Printf("[mTLS Whitelist] WARN: could not tighten %s (mode %o): its volume is mounted read-only; "+
					"the tampering risk is structurally removed by the mount, but the file content remains group/world-readable "+
					"— fix the source asset to ship it at 0600", dw.path, perm)
			} else {
				return fmt.Errorf("tighten whitelist file permissions: %w", chmodErr)
			}
		}
	}

	data, err := os.ReadFile(dw.path)
	if err != nil {
		return err
	}
	var conf WhitelistConfig
	if err := func() error {
		decoder := yaml.NewDecoder(strings.NewReader(string(data)))
		decoder.KnownFields(true)
		return decoder.Decode(&conf)
	}(); err != nil {
		return err
	}

	newClients := make(map[string][]string)
	newCertFPs := make(map[string][]string)

	// Priority 1: "clients" key (design doc standard) / 优先使用 "clients" 键
	for _, c := range conf.Clients {
		if c.Enabled != nil && !*c.Enabled {
			continue
		}
		newClients[c.CN] = c.AllowedScopes
		if len(c.CertFingerprints) > 0 {
			newCertFPs[c.CN] = normalizeFingerprintList(c.CertFingerprints)
		}
	}

	sum := sha256.Sum256(data)
	fp := whitelistFingerprint{
		modTime: lstatInfo.ModTime(),
		size:    lstatInfo.Size(),
		hash:    hex.EncodeToString(sum[:]),
	}

	dw.mu.Lock()
	defer dw.mu.Unlock()
	if len(newClients) == 0 && len(dw.clients) > 0 {
		// 解析结果为空且现有配置非空：拒绝替换（保旧配置），防止「误清空文件 / 攻击者
		// 覆写为空」导致全部客户端被静默放行。指纹照常更新，避免后台轮询每周期重复报错。
		log.Printf("[mTLS Whitelist] ERROR: reload of %s produced zero authorized CN entries while %d existing entries are active; keeping previous configuration", dw.path, len(dw.clients))
		dw.fingerprint = fp
		return nil
	}
	dw.clients = newClients
	dw.certFPs = newCertFPs
	dw.defaultScopes = conf.DefaultScopes
	dw.fingerprint = fp
	log.Printf("[mTLS Whitelist] Reloaded %d authorized CN entries from %s", len(newClients), dw.path)
	return nil
}

// CheckReload checks whether the whitelist file on disk was modified, and reloads if so.
//
// CheckReload 在 gRPC 拦截器热路径上执行增量检测：
//   - 廉价路径：Lstat 的 mtime + size 任一变化即触发 reload（覆盖正常编辑与绝大多数篡改）；
//   - 内容 hash 抽检：mtime/size 未变时每 2 秒至多一次读取并比对 sha256，
//     捕获「保留 mtime 且等尺寸」的精心篡改；符号链接文件一律视为无变化。
func (dw *DynamicWhitelist) CheckReload() bool {
	if dw.path == "" {
		return false
	}
	info, err := os.Lstat(dw.path)
	if err != nil || info.Mode()&os.ModeSymlink != 0 {
		return false
	}
	dw.mu.RLock()
	fp := dw.fingerprint
	dw.mu.RUnlock()

	if info.ModTime().After(fp.modTime) || info.Size() != fp.size {
		return dw.reload() == nil
	}

	now := time.Now()
	last := time.Unix(0, dw.lastHashCheck.Load())
	if now.Sub(last) < whitelistHashThrottle {
		return false
	}
	data, err := os.ReadFile(dw.path)
	if err != nil {
		return false
	}
	dw.lastHashCheck.Store(now.UnixNano())
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]) != fp.hash && dw.reload() == nil
}

// DefaultScopes returns a copy of configured default scopes.
func (dw *DynamicWhitelist) DefaultScopes() []string {
	dw.mu.RLock()
	defer dw.mu.RUnlock()
	if dw.defaultScopes == nil {
		return nil
	}
	res := make([]string, len(dw.defaultScopes))
	copy(res, dw.defaultScopes)
	return res
}

// poll watches for file changes by polling modification time.
//
// poll 以后台协程方式运行，每隔 5 秒调用 CheckReload 执行 mtime + size + 节流内容 hash 检测：
// - 任一指纹维度变化触发 reload（详情见 CheckReload）；
// - 接收到 stopCh 信号时退出循环。
func (dw *DynamicWhitelist) poll() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if dw.CheckReload() {
				continue
			}
			// reload 失败（如临时读写竞争）不重试直到下个周期，错误已在 reload 内记录
		case <-dw.stopCh:
			return
		}
	}
}

// Close stops the background file polling goroutine.
//
// Close 安全停止后台轮询协程，支持多次重复调用（幂等保护）。
func (dw *DynamicWhitelist) Close() {
	dw.stopMu.Lock()
	defer dw.stopMu.Unlock()
	if !dw.stopped {
		close(dw.stopCh)
		dw.stopped = true
	}
}

// IsAuthorized checks whether a client CN is present in the whitelist.
//
// IsAuthorized 快速查询客户端证书 CN 是否登记在白名单中（在读锁保护下执行）。
func (dw *DynamicWhitelist) IsAuthorized(clientCN string) bool {
	dw.mu.RLock()
	defer dw.mu.RUnlock()
	_, exists := dw.clients[clientCN]
	return exists
}

// CheckScope checks whether a client CN is authorized for a specific method/scope.
//
// CheckScope 检查客户端 CN 是否被授权访问指定的 gRPC 方法全名：
//
// ==============================================================================
// 【Scope 匹配规则】
// 1. 【全局通配符 ("*")】：若配置包含 `*`，允许访问所有方法；
// 2. 【精确全名匹配 (Exact Match)】：配置字符串与 method 完全一致（如 `/PrivacyService/Process`）；
// 3. 【前缀通配符 (Prefix Wildcard)】：如 `/AuditLog/*` 可匹配 `/AuditLog/RecordAudit`、`/AuditLog/ListLogs` 等；
// 4. 【返回值】：(authorized: 是否通过, scopes: 该 CN 拥有的全部 scope 列表)。
// ==============================================================================
func (dw *DynamicWhitelist) CheckScope(clientCN string, method string) (bool, []string) {
	dw.mu.RLock()
	defer dw.mu.RUnlock()

	scopes, exists := dw.clients[clientCN]
	if !exists {
		return false, nil
	}

	for _, s := range scopes {
		if s == "*" || s == method || matchScopePattern(s, method) {
			return true, scopes
		}
	}
	return false, scopes
}

// GetScopes returns the allowed scopes for a given CN.
//
// GetScopes 返回指定客户端 CN 的所有已授权 Scope 列表副本。
func (dw *DynamicWhitelist) GetScopes(clientCN string) ([]string, bool) {
	dw.mu.RLock()
	defer dw.mu.RUnlock()
	scopes, exists := dw.clients[clientCN]
	return scopes, exists
}

// GetCertFingerprints 返回指定 CN 配置的证书指纹钉扎列表副本。
// 第二返回值表示是否启用钉扎：false（或空列表）表示保持既有 CN 匹配语义不校验指纹。
func (dw *DynamicWhitelist) GetCertFingerprints(clientCN string) ([]string, bool) {
	dw.mu.RLock()
	defer dw.mu.RUnlock()
	fps, pinned := dw.certFPs[clientCN]
	if !pinned || len(fps) == 0 {
		return nil, false
	}
	return append([]string(nil), fps...), true
}

// normalizeFingerprintList 归一化指纹列表：去冒号/空格、统一小写、去重、剔除非法项。
func normalizeFingerprintList(fps []string) []string {
	seen := make(map[string]struct{}, len(fps))
	out := make([]string, 0, len(fps))
	for _, raw := range fps {
		cleaned := strings.ToLower(strings.NewReplacer(":", "", " ", "").Replace(strings.TrimSpace(raw)))
		if len(cleaned) != 64 {
			continue
		}
		if _, dup := seen[cleaned]; dup {
			continue
		}
		seen[cleaned] = struct{}{}
		out = append(out, cleaned)
	}
	return out
}

// AuthorizedClients returns a copy of the authorized CN to scopes mapping.
func (dw *DynamicWhitelist) AuthorizedClients() map[string][]string {
	dw.mu.RLock()
	defer dw.mu.RUnlock()
	res := make(map[string][]string, len(dw.clients))
	for k, v := range dw.clients {
		scopesCopy := make([]string, len(v))
		copy(scopesCopy, v)
		res[k] = scopesCopy
	}
	return res
}

// matchScopePattern performs simple wildcard pattern matching.
//
// matchScopePattern 针对通配符模式执行字符串匹配：
// - `*`：匹配任意方法；
// - `/ServiceHub/*`：匹配所有以 `/ServiceHub/` 为前缀的方法名；
// - 其他：执行精确字符串等值比对。
func matchScopePattern(pattern, value string) bool {
	if pattern == "*" {
		return true
	}
	// 对 "/ServiceHub/*" 等模式执行高效前缀匹配
	if len(pattern) > 2 && pattern[len(pattern)-1] == '*' && pattern[len(pattern)-2] == '/' {
		prefix := pattern[:len(pattern)-1] // 提取 "/ServiceHub/"
		return len(value) >= len(prefix) && value[:len(prefix)] == prefix
	}
	return pattern == value
}
