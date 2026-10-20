// Package config 提供共享的环境变量与配置文件加载助手。
package config

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

// LoadDotEnv 从指定路径列表加载 .env 配置文件。
//
// 解析规则：
//  1. 忽略空行与以 '#' 开头的注释行；
//  2. 按第一个 '=' 切割为 key 和 value；
//  3. 支持剥离单引号 ('...') 与双引号 ("...")；
//  4. 遵从「显式系统环境变量优先」原则：若当前进程中已存在该环境变量（通过 os.LookupEnv），
//     则不会被 .env 文件中的值覆盖，保证命令行传参及容器平台注入的最高优先级。
//
// 返回成功加载并注入新环境变量的数量，以及任何非文件不存在的读取错误。
func LoadDotEnv(paths ...string) (int, error) {
	totalSet := 0
	for _, p := range paths {
		if p == "" {
			continue
		}
		f, err := os.Open(p)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return totalSet, err
		}

		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			// 忽略空行与注释行
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			idx := strings.IndexByte(line, '=')
			if idx <= 0 {
				continue
			}
			key := strings.TrimSpace(line[:idx])
			val := strings.TrimSpace(line[idx+1:])

			// 剥离两端包裹的引号
			if len(val) >= 2 && ((val[0] == '"' && val[len(val)-1] == '"') || (val[0] == '\'' && val[len(val)-1] == '\'')) {
				val = val[1 : len(val)-1]
			} else {
				// 未加引号时，剥离行尾行内注释（如 FOO=bar # comment）
				if commentIdx := strings.Index(val, " #"); commentIdx >= 0 {
					val = strings.TrimSpace(val[:commentIdx])
				}
			}

			// 显式系统环境变量优先：只要该变量在本进程里**被设置过**（哪怕是空串或纯空白），
			// .env 都不许覆盖。判据必须是 os.LookupEnv 的存在性，而不是 Getenv 的非空性 ——
			// 后者会把 `API_KEY=`（容器编排里「我就是要它为空」）当成「没设过」，
			// 让 .env 里的开发密钥静默填进去，正好绕过启动期的 fail-closed 告警。
			if _, ok := os.LookupEnv(key); !ok {
				if err := os.Setenv(key, val); err == nil {
					totalSet++
				}
			}
		}
		_ = f.Close()
		if err := scanner.Err(); err != nil {
			return totalSet, err
		}
	}
	return totalSet, nil
}

// LoadDotEnvAuto 自动探测并加载就近的 .env 配置文件。
//
// 探测顺序：
//  1. DOTENV_PATH 环境变量指定的自定义路径（若存在）；
//  2. 当前工作目录下的 .env；
//  3. 向上逐级探查父目录（../.env, ../../.env, ../../../.env），适配从子目录运行 go run 的开发形态。
func LoadDotEnvAuto() {
	var candidates []string
	if custom := strings.TrimSpace(os.Getenv("DOTENV_PATH")); custom != "" {
		candidates = append(candidates, custom)
	}
	candidates = append(candidates, ".env", "../.env", "../../.env", "../../../.env")

	for _, c := range candidates {
		if abs, err := filepath.Abs(c); err == nil {
			if info, err := os.Stat(abs); err == nil && !info.IsDir() {
				_, _ = LoadDotEnv(abs)
				return // 命中并加载最近的 .env 即收敛，避免重复覆盖
			}
		}
	}
}
