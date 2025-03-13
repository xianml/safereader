package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"log"
	"strings"
	"sort"
)

// 定义 Safetensors 文件中的 Metadata 结构
type Header struct {
    Metadata *MetadataInfo                `json:"__metadata__,omitempty"`
    Tensors  map[string]*TensorInfo       `json:"-"`
}

type MetadataInfo struct {
    Format   string            `json:"format"`
    Metadata map[string]string `json:"metadata"`
}

type TensorInfo struct {
    Dtype       string    `json:"dtype"`
    Shape       []int64   `json:"shape"`
    DataOffsets []uint64  `json:"data_offsets"`
}

func parseHeader(data []byte) (*Header, error) {
    var raw map[string]json.RawMessage
    if err := json.Unmarshal(data, &raw); err != nil {
        return nil, fmt.Errorf("failed to parse header: %w", err)
    }

    header := &Header{
        Tensors: make(map[string]*TensorInfo),
    }

    for key, value := range raw {
        if key == "__metadata__" {
            var meta MetadataInfo
            if err := json.Unmarshal(value, &meta); err != nil {
                return nil, fmt.Errorf("failed to parse metadata: %w", err)
            }
            header.Metadata = &meta
            continue
        }

        var tensor TensorInfo
        if err := json.Unmarshal(value, &tensor); err != nil {
            return nil, fmt.Errorf("failed to parse tensor info for %s: %w", key, err)
        }
        header.Tensors[key] = &tensor
    }

    return header, nil
}

// 读取 Safetensors 文件的 metadata
func readSafetensorsMetadata(filePath string) (*Header, error) {
	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// 读取文件头的前 8 字节，获取 header 大小
	var headerSize uint64
	if err := binary.Read(file, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("failed to read header size: %w", err)
	}

	// 读取 header 部分的 JSON 字符串
	headerData := make([]byte, headerSize)
	if _, err := io.ReadFull(file, headerData); err != nil {
		return nil, fmt.Errorf("failed to read header data: %w", err)
	}

	// 解析 JSON header
	header, err := parseHeader(headerData)
	if err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	return header, nil
}

func main() {
	// 从命令行获取文件路径作为参数
	if len(os.Args) < 2 {
		log.Fatal("Please provide the safetensors file path as the first argument.")
	}
	filePath := os.Args[1]

	// 读取并解析 Safetensors 文件的元数据
	header, err := readSafetensorsMetadata(filePath)
	if err != nil {
		log.Fatalf("Error reading safetensors metadata: %v", err)
	}

	// 首先输出 __metadata__ 信息（如果存在）
	if header.Metadata != nil {
		fmt.Println("\n------------------------ Metadata Information ------------------------")
		
		metadataJsonStr, err := json.MarshalIndent(header.Metadata, "", "  ")
		if err != nil {
			log.Fatalf("Error marshalling metadata JSON: %v", err)
		}
		fmt.Println(string(metadataJsonStr))
	}

	// 输出张量信息
	fmt.Println("\n------------------------ Tensor Information -------------------------")
	fmt.Printf("%-40s %-8s %-20s %-20s\n", "Tensor Name", "Dtype", "Shape", "Data Offsets")
	fmt.Println(strings.Repeat("-", 92))

	// 收集并排序张量名称
	tensorNames := make([]string, 0, len(header.Tensors))
	for name := range header.Tensors {
		tensorNames = append(tensorNames, name)
	}
	sort.Strings(tensorNames)

	// 按排序后的顺序输出张量信息
	for _, tensorName := range tensorNames {
		metadata := header.Tensors[tensorName]
		
		// 格式化 shape
		shapeStr := fmt.Sprintf("[%d", metadata.Shape[0])
		for _, dim := range metadata.Shape[1:] {
			shapeStr += fmt.Sprintf(",%d", dim)
		}
		shapeStr += "]"
		
		// 格式化 data offsets
		offsetStr := fmt.Sprintf("[%d-%d]", metadata.DataOffsets[0], metadata.DataOffsets[1])
		
		// 如果张量名称过长，进行截断处理
		displayName := tensorName
		if len(tensorName) > 37 {
			displayName = tensorName[:34] + "..."
		}
		
		fmt.Printf("%-40s %-8s %-20s %-20s\n", 
			displayName, 
			metadata.Dtype, 
			shapeStr, 
			offsetStr)
	}
}
