package main

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
	BlockSize    = 4096 // 4KB数据块
	SubBlockSize = 16   // 16B子块
	HashSize     = 16   // SHA128输出长度
	KeySize      = 32   // AES-256密钥长度
	MinHashSize  = 64   // MinHash签名长度
)

// 加密方案枚举
type EncryptionMethod int

const (
	StandardAES EncryptionMethod = iota
	HashDerivedKey
	MinHashDerived
)

// 块类型定义
type BlockType int

const (
	UniqueBlock BlockType = iota
	DuplicateBlock
)

// 块数据结构
type DataBlock struct {
	Type     BlockType
	Index    int
	Hash     [HashSize]byte
	Content  []byte
	Cipher   []byte
	RefIndex int // 对于重复块，引用的原始块索引
}

// 性能统计结构
type PerformanceMetrics struct {
	KeyGenTime      time.Duration
	EncryptTime     time.Duration
	DecryptTime     time.Duration
	MemAlloc        uint64
	UniqueBlocks    int
	DuplicateBlocks int
	DedupRatio      float64
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

func newMinHashComputer(numBlocks int) *minHashComputer {
	return &minHashComputer{
		lsh:        NewMinhashLSH16(MinHashSize, 0.4, numBlocks*2),
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
func generateDerivedKey(blocks []*DataBlock) [KeySize]byte {
	var combinedHash [HashSize]byte

	for _, block := range blocks {
		for i := range combinedHash {
			combinedHash[i] ^= block.Hash[i]
		}
	}
	return sha256.Sum256(combinedHash[:])
}

// 生成密钥 (方案3: MinHash推导)
func generateMinHashKey(blocks []*DataBlock) [KeySize]byte {
	minhash := newMinHashComputer(len(blocks))
	for i, block := range blocks {
		minhash.addBlock(i, block.Content)
	}
	combinedHash := minhash.finalize()
	return sha256.Sum256(combinedHash[:])
}

// CTR模式加密/解密核心 (修正版)
func processCTR(data []byte, key [KeySize]byte, blockHash [HashSize]byte, blockIndex int) []byte {
	block, _ := aes.NewCipher(key[:])

	// 创建足够空间的计数器 (HashSize + 8字节索引)
	counter := make([]byte, HashSize+8)

	// 复制块哈希到计数器前部
	copy(counter[:HashSize], blockHash[:])

	// 安全写入索引到计数器后部
	binary.BigEndian.PutUint64(counter[HashSize:], uint64(blockIndex))

	// 使用前16字节作为CTR模式的IV (保持原有IV生成方式)
	iv := counter[:aes.BlockSize]

	ctr := cipher.NewCTR(block, iv)
	result := make([]byte, len(data))
	ctr.XORKeyStream(result, data)
	return result
}

// 完整加密流程（含重复数据删除）
func encryptData(data []byte, method EncryptionMethod) ([KeySize]byte, []*DataBlock, PerformanceMetrics) {
	var metrics PerformanceMetrics
	startTime := time.Now()

	// 分块处理
	numBlocks := (len(data) + BlockSize - 1) / BlockSize
	blocks := make([]*DataBlock, numBlocks)

	// 哈希映射表用于检测重复块
	hashMap := make(map[[HashSize]byte]int)

	// 初始化块结构
	for i := range blocks {
		start := i * BlockSize
		end := start + BlockSize
		if end > len(data) {
			end = len(data)
		}
		content := data[start:end]
		hash := sha128(content)

		blocks[i] = &DataBlock{
			Index:   i,
			Content: content,
			Hash:    hash,
		}
	}

	// 检测重复块
	for i, block := range blocks {
		if refIndex, exists := hashMap[block.Hash]; exists {
			block.Type = DuplicateBlock
			block.RefIndex = refIndex
			metrics.DuplicateBlocks++
		} else {
			block.Type = UniqueBlock
			hashMap[block.Hash] = i
			metrics.UniqueBlocks++
		}
	}
	metrics.DedupRatio = float64(metrics.DuplicateBlocks) / float64(numBlocks)

	// 密钥生成
	var key [KeySize]byte
	switch method {
	case StandardAES:
		key = generateRandomKey()
	case HashDerivedKey:
		key = generateDerivedKey(blocks)
	case MinHashDerived:
		key = generateMinHashKey(blocks)
	}
	metrics.KeyGenTime = time.Since(startTime)

	// 加密处理（只加密唯一块）
	encStart := time.Now()
	for _, block := range blocks {
		if block.Type == UniqueBlock {
			block.Cipher = processCTR(block.Content, key, block.Hash, block.Index)
		}
	}
	metrics.EncryptTime = time.Since(encStart)

	// 内存统计
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	metrics.MemAlloc = m.Alloc

	return key, blocks, metrics
}

// 解密流程（处理重复块引用）
func decryptData(blocks []*DataBlock, key [KeySize]byte) ([]byte, PerformanceMetrics) {
	var metrics PerformanceMetrics
	decStart := time.Now()

	// 创建结果缓冲区
	totalSize := 0
	for _, block := range blocks {
		totalSize += len(block.Content)
	}
	result := make([]byte, totalSize)
	currentPos := 0

	// 解密块并处理重复引用
	for _, block := range blocks {
		var decrypted []byte

		switch block.Type {
		case UniqueBlock:
			// 使用块哈希进行解密（保持与加密时相同的IV生成方式）
			decrypted = processCTR(block.Cipher, key, block.Hash, block.Index)

		case DuplicateBlock:
			// 获取引用的原始块
			refBlock := blocks[block.RefIndex]
			if refBlock.Type != UniqueBlock {
				panic("invalid reference block")
			}

			// 直接复制原始块内容
			decrypted = make([]byte, len(block.Content))
			copy(decrypted, refBlock.Content)
		}

		// 复制到结果缓冲区
		copy(result[currentPos:], decrypted)
		currentPos += len(decrypted)
	}

	metrics.DecryptTime = time.Since(decStart)
	return result, metrics
}

// 性能测试主函数
func runPerformanceTest(data []byte, method EncryptionMethod) PerformanceMetrics {
	var metrics PerformanceMetrics

	// 加密阶段
	key, blocks, encMetrics := encryptData(data, method)
	metrics = encMetrics

	// 解密阶段
	decResult, decMetrics := decryptData(blocks, key)
	metrics.DecryptTime = decMetrics.DecryptTime

	// 验证解密正确性
	if len(data) != len(decResult) {
		panic(fmt.Sprintf("decryption size mismatch: %d vs %d", len(data), len(decResult)))
	}
	for i := range data {
		if data[i] != decResult[i] {
			panic(fmt.Sprintf("decryption content mismatch at position %d", i))
		}
	}

	return metrics
}

func main() {
	// 测试数据大小配置
	testSizes := []int{
		1 * 1024 * 1024,    // 1MB
		4 * 1024 * 1024,    // 4MB
		16 * 1024 * 1024,   // 16MB
		64 * 1024 * 1024,   // 64MB
		256 * 1024 * 1024,  // 256MB
		1024 * 1024 * 1024, // 1024MB
		4096 * 1024 * 1024, // 4096MB
	}

	for _, size := range testSizes {
		fmt.Printf("\n===== 测试数据大小: %.2f MB =====\n", float64(size)/1024/1024)

		// 生成测试数据（包含重复块）
		testData := generateTestData(size)

		// 运行三种方案测试
		methods := []struct {
			name string
			typ  EncryptionMethod
		}{
			{"标准AES-CTR", StandardAES},
			{"哈希派生密钥", HashDerivedKey},
			{"MinHash派生", MinHashDerived},
		}

		for _, m := range methods {
			fmt.Printf("\n--- %s 测试 ---\n", m.name)
			start := time.Now()
			metrics := runPerformanceTest(testData, m.typ)
			totalTime := time.Since(start)

			fmt.Printf("唯一块数量: %d\n", metrics.UniqueBlocks)
			fmt.Printf("重复块数量: %d\n", metrics.DuplicateBlocks)
			fmt.Printf("重复率: %.2f%%\n", metrics.DedupRatio*100)
			fmt.Printf("密钥生成耗时: %v\n", metrics.KeyGenTime)
			fmt.Printf("加密耗时: %v\n", metrics.EncryptTime)
			fmt.Printf("解密耗时: %v\n", metrics.DecryptTime)
			fmt.Printf("内存分配: %.2f MB\n", float64(metrics.MemAlloc)/1024/1024)
			fmt.Printf("总执行时间: %v\n", totalTime)
		}
	}
}

// 生成包含重复块的测试数据
func generateTestData(size int) []byte {
	data := make([]byte, size)
	rand.Read(data)

	// 创建重复模式（每10个块重复一次）
	blockSize := BlockSize
	if size < blockSize*20 {
		blockSize = size / 20
	}

	for i := 0; i < size/(blockSize*2); i++ {
		srcPos := i * blockSize
		dstPos := (i*2 + 1) * blockSize

		if dstPos+blockSize > size {
			break
		}

		// 复制块创建重复
		copy(data[dstPos:dstPos+blockSize], data[srcPos:srcPos+blockSize])
	}

	return data
}
