package main-V2

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sort"
	"time"
)

const (
	BlockSize   = 4096 // 4KB数据块
	HashSize    = 16   // SHA128输出长度
	KeySize     = 32   // AES-256密钥长度
	MinHashSize = 64   // MinHash签名长度
	TestRepeats = 3    // 每种配置测试次数
)

// 加密方案枚举
type EncryptionMethod int

const (
	StandardAES EncryptionMethod = iota
	HashDerivedKey
	MinHashDerived
)

// 性能统计结构
type PerformanceMetrics struct {
	TotalTime        time.Duration
	KeyGenTime       time.Duration
	HashGenTime      time.Duration // minHash专用：哈希生成时间
	EncryptTime      time.Duration
	DecryptTime      time.Duration
	MemAlloc         uint64
	UniqueBlocks     int
	DuplicateBlocks  int
	DedupRatio       float64
	SkippedEncrypts  int    // minHash专用：跳过的加密操作数
	FirstDupDetected int    // 首次检测到重复的块索引
	TestConfig       string // 测试配置描述
}

// ==================== MinHash LSH 实现 ====================
const (
	integrationPrecision = 0.01
)

type hashKeyFunc func([]uint64) string

func hashKeyFuncGen(hashValueSize int) hashKeyFunc {
	return func(sig []uint64) string {
		s := make([]byte, hashValueSize*len(sig))
		buf := make([]byte, 8)
		for i, v := range sig {
			binary.LittleEndian.PutUint64(buf, v)
			copy(s[i*hashValueSize:(i+1)*hashValueSize], buf[:hashValueSize])
		}
		return string(s)
	}
}

// Compute the integral of function f
func integral(f func(float64) float64, a, b, precision float64) float64 {
	var area float64
	for x := a; x < b; x += precision {
		area += f(x+0.5*precision) * precision
	}
	return area
}

// Probability density function for false positive
func falsePositive(l, k int) func(float64) float64 {
	return func(j float64) float64 {
		return 1.0 - math.Pow(1.0-math.Pow(j, float64(k)), float64(l))
	}
}

// Probability density function for false negative
func falseNegative(l, k int) func(float64) float64 {
	return func(j float64) float64 {
		return 1.0 - (1.0 - math.Pow(1.0-math.Pow(j, float64(k)), float64(l)))
	}
}

// Compute the cummulative probability of false negative
func probFalseNegative(l, k int, t, precision float64) float64 {
	return integral(falseNegative(l, k), t, 1.0, precision)
}

// Compute the cummulative probability of false positive
func probFalsePositive(l, k int, t, precision float64) float64 {
	return integral(falsePositive(l, k), 0, t, precision)
}

// optimalKL returns the optimal K and L
func optimalKL(numHash int, t float64) (optK, optL int, fp, fn float64) {
	minError := math.MaxFloat64
	for l := 1; l <= numHash; l++ {
		for k := 1; k <= numHash; k++ {
			if l*k > numHash {
				continue
			}
			currFp := probFalsePositive(l, k, t, integrationPrecision)
			currFn := probFalseNegative(l, k, t, integrationPrecision)
			currErr := currFn + currFp
			if minError > currErr {
				minError = currErr
				optK = k
				optL = l
				fp = currFp
				fn = currFn
			}
		}
	}
	return
}

// entry contains the hash key
type entry struct {
	hashKey string
	key     interface{}
}

// hashTable is a look-up table
type hashTable []entry

func (h hashTable) Len() int           { return len(h) }
func (h hashTable) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h hashTable) Less(i, j int) bool { return h[i].hashKey < h[j].hashKey }

// MinhashLSH represents a MinHash LSH
type MinhashLSH struct {
	k              int
	l              int
	hashTables     []hashTable
	hashKeyFunc    hashKeyFunc
	hashValueSize  int
	numIndexedKeys int
}

func newMinhashLSH(threshold float64, numHash, hashValueSize, initSize int) *MinhashLSH {
	k, l, _, _ := optimalKL(numHash, threshold)
	hashTables := make([]hashTable, l)
	for i := range hashTables {
		hashTables[i] = make(hashTable, 0, initSize)
	}
	return &MinhashLSH{
		k:              k,
		l:              l,
		hashValueSize:  hashValueSize,
		hashTables:     hashTables,
		hashKeyFunc:    hashKeyFuncGen(hashValueSize),
		numIndexedKeys: 0,
	}
}

// NewMinhashLSH16 uses 16-bit hash values
func NewMinhashLSH16(numHash int, threshold float64, initSize int) *MinhashLSH {
	return newMinhashLSH(threshold, numHash, 2, initSize)
}

func (f *MinhashLSH) Params() (k, l int) {
	return f.k, f.l
}

func (f *MinhashLSH) hashKeys(sig []uint64) []string {
	hs := make([]string, f.l)
	for i := 0; i < f.l; i++ {
		hs[i] = f.hashKeyFunc(sig[i*f.k : (i+1)*f.k])
	}
	return hs
}

// Add a key with MinHash signature
func (f *MinhashLSH) Add(key interface{}, sig []uint64) {
	hs := f.hashKeys(sig)
	for i := range f.hashTables {
		f.hashTables[i] = append(f.hashTables[i], entry{hs[i], key})
	}
}

// Index makes all keys searchable
func (f *MinhashLSH) Index() {
	for i := range f.hashTables {
		sort.Sort(f.hashTables[i])
	}
	f.numIndexedKeys = len(f.hashTables[0])
}

// Query returns candidate keys
func (f *MinhashLSH) Query(sig []uint64) []interface{} {
	set := f.query(sig)
	results := make([]interface{}, 0, len(set))
	for key := range set {
		results = append(results, key)
	}
	return results
}

func (f *MinhashLSH) query(sig []uint64) map[interface{}]bool {
	hashKeys := f.hashKeys(sig)
	results := make(map[interface{}]bool)
	for i := 0; i < f.l; i++ {
		hashTable := f.hashTables[i][:f.numIndexedKeys]
		hashKey := hashKeys[i]
		k := sort.Search(len(hashTable), func(x int) bool {
			return hashTable[x].hashKey >= hashKey
		})
		if k < len(hashTable) && hashTable[k].hashKey == hashKey {
			for j := k; j < len(hashTable) && hashTable[j].hashKey == hashKey; j++ {
				key := hashTable[j].key
				if _, exist := results[key]; !exist {
					results[key] = true
				}
			}
		}
	}
	return results
}

// ==================== 主加密框架 ====================

// SHA128计算 (SHA512/128)
func sha128(data []byte) [HashSize]byte {
	hash := sha512.Sum512(data)
	var result [HashSize]byte
	copy(result[:], hash[:HashSize])
	return result
}

// MinHash计算器
type minHashComputer struct {
	lsh         *MinhashLSH
	signatures  [][]uint64
	blocks      [][]byte
	initialized bool
}

func newMinHashComputer(numBlocks int, threshold float64) *minHashComputer {
	return &minHashComputer{
		lsh:        NewMinhashLSH16(MinHashSize, threshold, numBlocks*2),
		signatures: make([][]uint64, numBlocks),
		blocks:     make([][]byte, numBlocks),
	}
}

func (m *minHashComputer) addBlock(index int, block []byte) {
	if m.initialized {
		panic("MinHashComputer already finalized")
	}

	// 创建MinHash签名
	signature := make([]uint64, MinHashSize)
	for i := range signature {
		// 使用FNV哈希
		h := newFNV64a(uint64(i))
		h.Write(block)
		signature[i] = h.Sum64()
	}

	m.signatures[index] = signature
	m.blocks[index] = block
	m.lsh.Add(index, signature)
}

func (m *minHashComputer) finalize() [HashSize]byte {
	m.initialized = true
	m.lsh.Index() // 构建索引

	// 组合所有块的哈希
	var combinedHash [HashSize]byte

	for i := range m.signatures {
		// 查找相似块
		results := m.lsh.Query(m.signatures[i])
		blockHash := sha128(m.blocks[i])

		// 与相似块的哈希异或
		for _, idx := range results {
			if idxInt, ok := idx.(int); ok && idxInt != i {
				otherHash := sha128(m.blocks[idxInt])
				for j := range blockHash {
					blockHash[j] ^= otherHash[j]
				}
			}
		}

		// 累加到总哈希
		for j := range combinedHash {
			combinedHash[j] ^= blockHash[j]
		}
	}

	return combinedHash
}

// FNV64a哈希实现
type fnv64a uint64

func newFNV64a(seed uint64) *fnv64a {
	f := fnv64a(seed)
	return &f
}

func (f *fnv64a) Write(data []byte) {
	hash := uint64(*f)
	const prime64 = 1099511628211
	for _, b := range data {
		hash ^= uint64(b)
		hash *= prime64
	}
	*f = fnv64a(hash)
}

func (f fnv64a) Sum64() uint64 {
	return uint64(f)
}

// 生成密钥 (方案1: 随机密钥)
func generateRandomKey() [KeySize]byte {
	var key [KeySize]byte
	rand.Read(key[:])
	return key
}

// 生成密钥 (方案2: 哈希推导)
func generateDerivedKey(blocks [][]byte) [KeySize]byte {
	var combinedHash [HashSize]byte

	for _, block := range blocks {
		hash := sha128(block)
		for i := range combinedHash {
			combinedHash[i] ^= hash[i]
		}
	}
	return sha256.Sum256(combinedHash[:])
}

// 生成密钥 (方案3: MinHash推导)
func generateMinHashKey(blocks [][]byte, threshold float64) [KeySize]byte {
	minhash := newMinHashComputer(len(blocks), threshold)
	for i, block := range blocks {
		minhash.addBlock(i, block)
	}
	combinedHash := minhash.finalize()
	return sha256.Sum256(combinedHash[:])
}

// CTR模式加密/解密核心
func processCTR(data []byte, key [KeySize]byte, blockHash [HashSize]byte, blockIndex int) []byte {
	block, _ := aes.NewCipher(key[:])

	// 创建足够空间的计数器 (HashSize + 8字节索引)
	counter := make([]byte, HashSize+8)
	copy(counter[:HashSize], blockHash[:])
	binary.BigEndian.PutUint64(counter[HashSize:], uint64(blockIndex))

	// 使用前16字节作为CTR模式的IV
	iv := counter[:aes.BlockSize]

	ctr := cipher.NewCTR(block, iv)
	result := make([]byte, len(data))
	ctr.XORKeyStream(result, data)
	return result
}

// 完整加密流程（含重复数据删除）
func encryptData(data []byte, method EncryptionMethod, dupThreshold float64) ([KeySize]byte, [][]byte, PerformanceMetrics) {
	var metrics PerformanceMetrics

	// 分块处理
	numBlocks := (len(data) + BlockSize - 1) / BlockSize
	blocks := make([][]byte, numBlocks)

	// 初始化块结构
	for i := range blocks {
		start := i * BlockSize
		end := start + BlockSize
		if end > len(data) {
			end = len(data)
		}
		blocks[i] = data[start:end]
	}

	// 密钥生成
	var key [KeySize]byte
	switch method {
	case StandardAES:
		key = generateRandomKey()
	case HashDerivedKey:
		key = generateDerivedKey(blocks)
	case MinHashDerived:
		keyGenStart := time.Now()
		key = generateMinHashKey(blocks, dupThreshold)
		metrics.KeyGenTime = time.Since(keyGenStart)
	}

	// 加密处理
	encStart := time.Now()
	uniqueBlocks := make([][]byte, 0, numBlocks)
	blockMap := make(map[[HashSize]byte]int) // 块哈希到索引的映射
	firstDupDetected := -1

	for i, block := range blocks {
		blockHash := sha128(block)

		if method == MinHashDerived {
			if _, exists := blockMap[blockHash]; exists {
				metrics.DuplicateBlocks++
				if firstDupDetected == -1 {
					firstDupDetected = i
				}
				continue // 跳过重复块的加密
			}
			blockMap[blockHash] = i
		}

		// 加密唯一块
		_ = processCTR(block, key, blockHash, i)
		metrics.UniqueBlocks++
		uniqueBlocks = append(uniqueBlocks, block)
	}

	metrics.EncryptTime = time.Since(encStart)
	metrics.FirstDupDetected = firstDupDetected
	metrics.SkippedEncrypts = metrics.DuplicateBlocks
	metrics.DedupRatio = float64(metrics.DuplicateBlocks) / float64(numBlocks)

	// 内存统计
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	metrics.MemAlloc = m.Alloc

	return key, uniqueBlocks, metrics
}

// 性能测试主函数
func runPerformanceTest(data []byte, method EncryptionMethod, dupThreshold float64, configDesc string) PerformanceMetrics {
	startTime := time.Now()
	metrics := PerformanceMetrics{TestConfig: configDesc}

	// 加密阶段
	_, _, encMetrics := encryptData(data, method, dupThreshold)
	metrics.KeyGenTime = encMetrics.KeyGenTime
	metrics.HashGenTime = encMetrics.HashGenTime
	metrics.EncryptTime = encMetrics.EncryptTime
	metrics.MemAlloc = encMetrics.MemAlloc
	metrics.UniqueBlocks = encMetrics.UniqueBlocks
	metrics.DuplicateBlocks = encMetrics.DuplicateBlocks
	metrics.DedupRatio = encMetrics.DedupRatio
	metrics.SkippedEncrypts = encMetrics.SkippedEncrypts
	metrics.FirstDupDetected = encMetrics.FirstDupDetected

	// 注意：解密过程在此测试中不是重点，故省略
	metrics.TotalTime = time.Since(startTime)
	return metrics
}

// 生成测试数据（精确控制重复率）
func generateTestData(size int, dupRatio float64) []byte {
	totalBlocks := (size + BlockSize - 1) / BlockSize
	uniqueBlocks := int(float64(totalBlocks) * (1 - dupRatio))
	dupBlocks := totalBlocks - uniqueBlocks

	data := make([]byte, size)
	rand.Read(data) // 填充随机数据

	// 创建重复块
	for i := 0; i < dupBlocks; i++ {
		srcIdx := i % uniqueBlocks
		dstIdx := uniqueBlocks + i

		srcStart := srcIdx * BlockSize
		dstStart := dstIdx * BlockSize

		if dstStart+BlockSize > size {
			break
		}

		copy(data[dstStart:dstStart+BlockSize], data[srcStart:srcStart+BlockSize])
	}

	return data
}

func main() {
	// 测试数据大小
	testSize := 128 * 1024 * 1024 // 64MB

	// 测试不同重复率
	dupRatios := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	threshold := 0.1 // 固定检测阈值

	// 测试三种方案
	methods := []struct {
		name string
		typ  EncryptionMethod
	}{
		{"标准AES-CTR", StandardAES},
		{"哈希派生密钥", HashDerivedKey},
		{"MinHash派生", MinHashDerived},
	}

	fmt.Println("===== 重复数据加密性能测试 =====")
	fmt.Printf("数据大小: %.2f MB\n", float64(testSize)/1024/1024)
	fmt.Printf("检测阈值: %.1f\n", threshold)
	fmt.Println("===============================")

	// 表头
	fmt.Printf("%-20s | %-8s | %-10s | %-10s | %-10s | %-10s | %-10s | %-8s | %-12s\n",
		"方案", "重复率", "总时间(ms)", "密钥生成", "哈希生成", "加密时间", "跳过的加密", "内存(MB)", "首重复块")
	fmt.Println("-------------------------------------------------------------------------------------------------")

	for _, ratio := range dupRatios {
		testData := generateTestData(testSize, ratio)

		for _, m := range methods {
			var totalMetrics PerformanceMetrics

			// 多次测试取平均值
			for i := 0; i < TestRepeats; i++ {
				config := fmt.Sprintf("重复率:%.1f", ratio)
				metrics := runPerformanceTest(testData, m.typ, threshold, config)

				totalMetrics.TotalTime += metrics.TotalTime
				totalMetrics.KeyGenTime += metrics.KeyGenTime
				totalMetrics.HashGenTime += metrics.HashGenTime
				totalMetrics.EncryptTime += metrics.EncryptTime
				totalMetrics.MemAlloc += metrics.MemAlloc
				totalMetrics.SkippedEncrypts += metrics.SkippedEncrypts

				// 这些值在多次测试中相同，取最后一次
				totalMetrics.UniqueBlocks = metrics.UniqueBlocks
				totalMetrics.DuplicateBlocks = metrics.DuplicateBlocks
				totalMetrics.DedupRatio = metrics.DedupRatio
				totalMetrics.FirstDupDetected = metrics.FirstDupDetected
			}

			// 计算平均值
			avgTotal := totalMetrics.TotalTime / time.Duration(TestRepeats)
			avgKeyGen := totalMetrics.KeyGenTime / time.Duration(TestRepeats)
			avgHashGen := totalMetrics.HashGenTime / time.Duration(TestRepeats)
			avgEncrypt := totalMetrics.EncryptTime / time.Duration(TestRepeats)
			avgMem := float64(totalMetrics.MemAlloc/uint64(TestRepeats)) / 1024 / 1024
			avgSkipped := totalMetrics.SkippedEncrypts / TestRepeats

			// 输出结果
			fmt.Printf("%-20s | %-8.1f | %-10.2f | %-10.2f | %-10.2f | %-10.2f | %-10d | %-8.2f | %-12d\n",
				m.name,
				ratio,
				float64(avgTotal.Milliseconds()),
				float64(avgKeyGen.Milliseconds()),
				float64(avgHashGen.Milliseconds()),
				float64(avgEncrypt.Milliseconds()),
				avgSkipped,
				avgMem,
				totalMetrics.FirstDupDetected)
		}
		fmt.Println("-------------------------------------------------------------------------------------------------")
	}
}
