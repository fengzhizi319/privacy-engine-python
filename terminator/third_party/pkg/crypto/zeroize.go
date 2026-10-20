package crypto

import (
	"runtime"
)

// Zeroize 安全清零内存中的敏感字节切片（如对称密钥、明文私钥、派生凭据），
// 防范垃圾回收延迟导致的堆内存驻留泄露风险（密评 GM/T 0115 密钥生存期与销毁要求）。
func Zeroize(b []byte) {
	if len(b) == 0 {
		return
	}
	for i := range b {
		b[i] = 0
	}
	// runtime.KeepAlive 防止编译器优化将显式清零写操作消除（Dead Store Elimination）
	runtime.KeepAlive(b)
}

// ZeroizeStrings 批量清零多个敏感字节切片。
func ZeroizeStrings(slices ...[]byte) {
	for _, s := range slices {
		Zeroize(s)
	}
}
