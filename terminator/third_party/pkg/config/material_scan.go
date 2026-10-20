package config

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
)

// MaterialScan 描述"如何从一份服务配置里把所有凭据/TLS 材料路径扫出来"。
//
// 之所以是反射扫描而不是各服务手抄一张变量表：手工枚举正是这条门禁要防的那类缺陷
// ——新增一个 `*_TLCP_SIGN_KEY_FILE` 忘记登记，它就永远不被检查，而门禁依旧全绿
// （"配了但不生效"的同族）。扫描把完整性交给结构体本身：字段名以 File 结尾即入检。
type MaterialScan struct {
	// ServiceName 用于报错定位（privshield-engine / service-hub / …）。
	ServiceName string
	// Config 是服务配置（结构体或其指针）。反射遍历其中**导出**的字符串字段。
	Config any
	// EnvNames 把字段名映射回真正的环境变量名，让报错指向运维该改的那一行；
	// 未登记的字段以字段名示证（仍能定位，只是少一跳）。
	EnvNames map[string]string
	// Extras 覆盖不在 Config 结构体上、却有仓内缺省的材料（env 名 → 路径），
	// 例如 pkg/auth 自行读取的 `PRIVACY_AUTH_SIGNING_KEY_FILE`。
	Extras map[string]string
	// NotCredential 显式豁免**名字以 File 结尾但不是凭据材料**的字段（字段名 → 理由）。
	// 理由必须非空：豁免没有理由就等于下一条"看起来检查过"的假覆盖。
	// 同一字段不得同时出现在 EnvNames 与 NotCredential 里——两份清单互相矛盾时
	// 必须硬失败，而不是让后写的那一份悄悄裁决。
	NotCredential map[string]string
}

// ScanMaterialsOutsideRepoCheckout 反射扫出全部材料路径后交给
// ValidateMaterialOutsideRepoCheckout 判定（窄口径生产标注，判据与文案见该函数注释）。
//
// 扫描自身的完整性错误（陈旧豁免项、空理由、字段与 Extras 撞名）**不受生产标注影响**，
// 在开发态与 CI 即报：这些是门禁写错了，不是部署错了，等到生产才红就等于从未红过。
func ScanMaterialsOutsideRepoCheckout(scan MaterialScan) error {
	materials := make(map[string]string, 16)
	usedEnvNames := make(map[string]bool, len(scan.EnvNames))
	usedExemptions := make(map[string]bool, len(scan.NotCredential))

	reasons := make([]string, 0, 4)
	reasons = append(reasons, scanFieldStrings(scan.Config, "", materials, scan.EnvNames, usedEnvNames,
		scan.NotCredential, usedExemptions)...)

	// 陈旧登记：字段改名/删除后遗留的条目会让"这张表覆盖了全部材料"的假设失效。
	for _, name := range sortedScanKeys(scan.EnvNames) {
		if !usedEnvNames[name] {
			reasons = append(reasons, fmt.Sprintf("EnvNames entry %q matches no %q-suffixed field in %T — "+
				"either the field was renamed (fix this map) or the material is no longer scanned (do not leave the entry behind)", name, materialFieldSuffix, scan.Config))
		}
	}
	for _, name := range sortedScanKeys(scan.NotCredential) {
		if !usedExemptions[name] {
			reasons = append(reasons, fmt.Sprintf("NotCredential entry %q matches no scanned field in %T — stale exemption", name, scan.Config))
		}
	}
	for _, name := range sortedScanKeys(scan.Extras) {
		if _, dup := materials[name]; dup {
			reasons = append(reasons, fmt.Sprintf("Extras entry %q is already produced by the scan of %T — drop the duplicate", name, scan.Config))
		}
		materials[name] = scan.Extras[name]
	}
	if len(reasons) > 0 {
		sort.Strings(reasons)
		return fmt.Errorf("%s: material scan is misconfigured: %s", scan.ServiceName, strings.Join(reasons, "; "))
	}

	return ValidateMaterialOutsideRepoCheckout(scan.ServiceName, materials)
}

// materialFieldSuffix 是入检字段的命名约定：本项目所有材料路径字段都以 File 结尾
// （*_CERT_FILE / *_KEY_FILE / *_CA_FILE / *_WHITELIST_FILE / *_STORE_FILE）。
const materialFieldSuffix = "File"

// scanFieldStrings 递归遍历 cfg（结构体或指针）里的字符串字段，把 materialFieldSuffix 结尾的
// 导出字段按 env 名写入 materials；返回扫描配置错误（与生产标注无关，恒检查）。
func scanFieldStrings(cfg any, path string, materials map[string]string, envNames map[string]string,
	usedEnvNames map[string]bool, notCredential map[string]string, usedExemptions map[string]bool) []string {
	var problems []string
	if cfg == nil {
		return problems
	}
	v := reflect.ValueOf(cfg)
	for depth := 0; depth < 4; depth++ {
		switch v.Kind() {
		case reflect.Pointer:
			if v.IsNil() {
				return problems
			}
			v = v.Elem()
			continue
		default:
		}
		break
	}
	if v.Kind() != reflect.Struct {
		return problems
	}
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		sf := t.Field(i)
		if sf.PkgPath != "" { // 未导出字段不属于启动配置面
			continue
		}
		fieldPath := sf.Name
		if path != "" {
			fieldPath = path + "." + sf.Name
		}
		fv := v.Field(i)
		if sf.Type.Kind() == reflect.Struct ||
			(sf.Type.Kind() == reflect.Pointer && sf.Type.Elem().Kind() == reflect.Struct) {
			if !fv.IsValid() || (fv.Kind() == reflect.Pointer && fv.IsNil()) {
				continue
			}
			problems = append(problems, scanFieldStrings(fv.Interface(), fieldPath, materials, envNames, usedEnvNames,
				notCredential, usedExemptions)...)
			continue
		}
		if sf.Type.Kind() != reflect.String || !strings.HasSuffix(sf.Name, materialFieldSuffix) {
			continue
		}
		if reason, exempted := notCredential[sf.Name]; exempted {
			usedExemptions[sf.Name] = true
			if strings.TrimSpace(reason) == "" {
				problems = append(problems, fmt.Sprintf("exempted material field %s has an empty reason — an exemption must say why it is not credential material", fieldPath))
			}
			if _, named := envNames[sf.Name]; named {
				// 同一字段既被豁免又被登记为材料 = 两份互相矛盾的清单，与路由审计的"双登记即硬失败"同口径：
				// 保留它只会让后写的那一份悄悄裁决，而看起来像是经过讨论的决定。
				problems = append(problems, fmt.Sprintf("material field %s is both exempted (NotCredential) and named (EnvNames) — remove one", fieldPath))
			}
			continue
		}
		env, ok := envNames[sf.Name]
		if !ok {
			env = sf.Name
		}
		if ok && strings.TrimSpace(env) == "" {
			problems = append(problems, fmt.Sprintf("EnvNames entry %q is blank — name the variable or delete the entry", sf.Name))
		}
		usedEnvNames[sf.Name] = true
		materials[env] = fv.String()
	}
	return problems
}

func sortedScanKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
