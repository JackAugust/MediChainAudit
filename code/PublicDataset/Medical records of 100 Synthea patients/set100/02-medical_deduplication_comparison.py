import os
import re
import time
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH

class BaseDeduplicator:
    """基础去重器类"""
    def __init__(self):
        self.records = {}
        self.duplicate_count = 0
        self.original_size = 0
        self.deduplicated_size = 0
        self.processing_time = 0
        self.patient_metrics = {}  # 存储每个患者的性能指标
    
    def parse_medical_record(self, content, patient_id):
        """解析医疗记录文件内容"""
        records = []
        current_record = None
        
        for line in content.split('\n'):
            if line.startswith('就诊 #'):
                if current_record:
                    records.append(current_record)
                current_record = {
                    'patient_id': patient_id,
                    'diagnoses': [], 
                    'procedures': [], 
                    'observations': [], 
                    'immunizations': [],
                    'medications': [],
                    'date': None
                }
                # 提取就诊编号
                visit_match = re.search(r'就诊 #(\d+):', line)
                if visit_match:
                    current_record['visit_id'] = visit_match.group(1)
                
                # 提取时间信息
                date_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        current_record['date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        current_record['date'] = None
            elif line.startswith('就诊类型:'):
                current_record['visit_type'] = line.split(':')[1].strip()
            elif line.startswith('描述:'):
                current_record['description'] = line.split(':')[1].strip()
            elif line.startswith('诊断:'):
                current_record['diagnoses'] = []
            elif line.startswith('程序/手术:'):
                current_record['procedures'] = []
            elif line.startswith('观察结果:'):
                current_record['observations'] = []
            elif line.startswith('免疫:'):
                current_record['immunizations'] = []
            elif line.startswith('药物:'):
                current_record['medications'] = []
            elif line.startswith('  - '):
                # 根据当前部分添加内容
                if 'diagnoses' in current_record and current_record['diagnoses'] is not None:
                    diagnosis = line[4:].split('(')[0].strip()
                    current_record['diagnoses'].append(diagnosis)
                elif 'procedures' in current_record and current_record['procedures'] is not None:
                    procedure = line[4:].split('(')[0].strip()
                    current_record['procedures'].append(procedure)
                elif 'observations' in current_record and current_record['observations'] is not None:
                    observation = line[4:].split(':')[0].strip()
                    current_record['observations'].append(observation)
                elif 'immunizations' in current_record and current_record['immunizations'] is not None:
                    immunization = line[4:].split('-')[0].strip()
                    current_record['immunizations'].append(immunization)
                elif 'medications' in current_record and current_record['medications'] is not None:
                    medication = line[4:].split('(')[0].strip()
                    current_record['medications'].append(medication)
    
        if current_record:
            records.append(current_record)
        
        return records

    def create_text_representation(self, record_data):
        """创建记录的文本表示用于哈希计算"""
        text_parts = []
        
        if 'diagnoses' in record_data and record_data['diagnoses']:
            text_parts.extend(record_data['diagnoses'])
        if 'description' in record_data:
            text_parts.append(record_data['description'])
        if 'visit_type' in record_data:
            text_parts.append(record_data['visit_type'])
        if 'procedures' in record_data and record_data['procedures']:
            text_parts.extend(record_data['procedures'][:3])  # 只取前3个过程
        
        # 添加时间信息以增加区分度
        if 'date' in record_data and record_data['date']:
            # 只使用年份和月份，避免过于具体的时间导致无法检测重复
            text_parts.append(f"year_{record_data['date'].year}")
            text_parts.append(f"month_{record_data['date'].month}")
        
        return ' '.join(text_parts)

    def calculate_storage_savings(self, input_folder, output_folder):
        """计算存储节省"""
        # 计算原始大小
        self.original_size = 0
        patient_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        for filename in patient_files:
            filepath = os.path.join(input_folder, filename)
            self.original_size += os.path.getsize(filepath)
        
        # 计算去重后大小
        self.deduplicated_size = 0
        deduplicated_files = [f for f in os.listdir(output_folder) if f.endswith('.txt')]
        for filename in deduplicated_files:
            filepath = os.path.join(output_folder, filename)
            self.deduplicated_size += os.path.getsize(filepath)
        
        return {
            "original_size_bytes": self.original_size,
            "deduplicated_size_bytes": self.deduplicated_size,
            "size_reduction_bytes": self.original_size - self.deduplicated_size,
            "size_reduction_percentage": (1 - self.deduplicated_size / self.original_size) * 100 if self.original_size > 0 else 0,
            "compression_ratio": self.original_size / self.deduplicated_size if self.deduplicated_size > 0 else 0
        }

    def save_deduplicated_records(self, unique_records, output_folder):
        """保存去重后的记录"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 按患者分组保存
        patients_data = {}
        for record in unique_records:
            patient_id = record['patient_id']
            if patient_id not in patients_data:
                patients_data[patient_id] = []
            patients_data[patient_id].append(record)
        
        # 保存到文件
        for patient_id, records in patients_data.items():
            output_path = os.path.join(output_folder, f"{patient_id}_deduplicated.txt")
            
            # 将记录转换为文本格式
            output_content = f"患者去重后记录报告 - {patient_id}\n"
            output_content += "=" * 50 + "\n\n"
            
            for i, record in enumerate(records):
                output_content += f"就诊记录 #{i+1}\n"
                output_content += "-" * 30 + "\n"
                
                if 'visit_type' in record:
                    output_content += f"就诊类型: {record['visit_type']}\n"
                if 'description' in record:
                    output_content += f"描述: {record['description']}\n"
                if 'date' in record and record['date']:
                    output_content += f"时间: {record['date']}\n"
                
                if record['diagnoses']:
                    output_content += "诊断:\n"
                    for diagnosis in record['diagnoses']:
                        output_content += f"  - {diagnosis}\n"
                
                output_content += "\n"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
        
        return len(patients_data)

    def calculate_patient_metrics(self, patient_id, original_records, unique_records, processing_time):
        """计算单个患者的性能指标"""
        # 计算记录数指标
        original_count = len(original_records)
        deduplicated_count = len(unique_records)
        duplicate_count = original_count - deduplicated_count
        
        # 计算大小指标（字节）
        original_size = sum(len(str(record)) for record in original_records)
        deduplicated_size = sum(len(str(record)) for record in unique_records)
        size_reduction = original_size - deduplicated_size
        size_reduction_percentage = (size_reduction / original_size * 100) if original_size > 0 else 0
        compression_ratio = original_size / deduplicated_size if deduplicated_size > 0 else 0
        
        # 存储指标
        self.patient_metrics[patient_id] = {
            "original_count": original_count,
            "deduplicated_count": deduplicated_count,
            "duplicate_count": duplicate_count,
            "processing_time": processing_time,
            "original_size_bytes": original_size,
            "deduplicated_size_bytes": deduplicated_size,
            "size_reduction_bytes": size_reduction,
            "size_reduction_percentage": size_reduction_percentage,
            "compression_ratio": compression_ratio
        }
        
        return self.patient_metrics[patient_id]

    def get_statistical_summary(self):
        """获取所有患者的统计摘要（均值和标准差）"""
        if not self.patient_metrics:
            return {}
        
        # 转换为DataFrame以便计算统计量
        df = pd.DataFrame.from_dict(self.patient_metrics, orient='index')
        
        # 计算均值和标准差
        summary = {}
        for column in df.columns:
            if df[column].dtype in [np.int64, np.float64]:
                summary[f"{column}_mean"] = df[column].mean()
                summary[f"{column}_std"] = df[column].std()
        
        return summary

class MLEDeduplicator(BaseDeduplicator):
    """基于MLE的确定性去重算法"""
    def __init__(self, similarity_threshold=0.98):  # 提高阈值，避免过度去重
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = None
        self.doc_ids = []
    
    def add_record(self, record_id, record_data):
        """添加记录并检查重复"""
        text_representation = self.create_text_representation(record_data)
        
        # 存储记录
        self.records[record_id] = {
            'data': record_data,
            'text': text_representation
        }
        
        # 如果是第一个记录，直接添加
        if not self.doc_ids:
            self.doc_ids.append(record_id)
            self.doc_vectors = self.vectorizer.fit_transform([text_representation])
            return True
        
        # 计算与所有已有记录的相似度
        current_vector = self.vectorizer.transform([text_representation])
        similarities = cosine_similarity(current_vector, self.doc_vectors)
        max_similarity = np.max(similarities)
        
        # 检查是否重复
        if max_similarity >= self.similarity_threshold:
            self.duplicate_count += 1
            return False
        else:
            # 添加到文档集合
            self.doc_ids.append(record_id)
            # 更新向量矩阵
            from scipy.sparse import vstack
            self.doc_vectors = vstack([self.doc_vectors, current_vector])
            return True

    def process_files(self, input_folder, output_folder):
        """处理所有文件"""
        start_time = time.time()
        
        patient_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        unique_records = []
        
        # 处理每个文件
        for filename in patient_files:
            filepath = os.path.join(input_folder, filename)
            patient_id = filename.replace('.txt', '')
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                records = self.parse_medical_record(content, patient_id)
                
                # 为每个记录生成唯一ID并处理
                patient_start_time = time.time()
                patient_unique_records = []
                
                for i, record in enumerate(records):
                    record_id = f"{patient_id}_{i}"
                    is_unique = self.add_record(record_id, record)
                    
                    if is_unique:
                        patient_unique_records.append(record)
                
                # 计算该患者的处理时间
                patient_processing_time = time.time() - patient_start_time
                
                # 计算该患者的性能指标
                self.calculate_patient_metrics(
                    patient_id, records, patient_unique_records, patient_processing_time
                )
                
                # 添加到总唯一记录列表
                unique_records.extend(patient_unique_records)
        
        # 保存去重后的记录
        self.save_deduplicated_records(unique_records, output_folder)
        
        # 计算存储节省
        storage_savings = self.calculate_storage_savings(input_folder, output_folder)
        
        self.processing_time = time.time() - start_time
        
        return {
            "unique_records": len(unique_records),
            "duplicate_count": self.duplicate_count,
            "processing_time": self.processing_time,
            "patient_metrics": self.patient_metrics,
            "statistical_summary": self.get_statistical_summary(),
            **storage_savings
        }

class FuzzyDeduplicator(BaseDeduplicator):
    """基于MinHash的模糊去重算法"""
    def __init__(self, threshold=0.4, num_perm=256):  # 降低阈值，增加置换次数
        super().__init__()
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = {}
    
    def create_minhash(self, text_data):
        """创建MinHash"""
        m = MinHash(num_perm=self.num_perm)
        words = text_data.split()
        
        for word in words:
            m.update(word.encode('utf8'))
        return m

    def add_record(self, record_id, record_data):
        """添加记录并检查重复"""
        text_representation = self.create_text_representation(record_data)
        
        # 存储记录
        self.records[record_id] = {
            'data': record_data,
            'text': text_representation
        }
        
        # 为当前记录创建MinHash
        current_mh = self.create_minhash(text_representation)
        self.minhashes[record_id] = current_mh
        
        # 检查是否有相似记录
        similar_ids = self.lsh.query(current_mh)
        is_duplicate = False
        
        for similar_id in similar_ids:
            if similar_id == record_id:
                continue
                
            # 计算实际相似度
            actual_similarity = current_mh.jaccard(self.minhashes[similar_id])
            
            if actual_similarity >= self.threshold:
                is_duplicate = True
                self.duplicate_count += 1
                break
        
        # 如果不是重复记录，添加到LSH索引
        if not is_duplicate:
            self.lsh.insert(record_id, current_mh)
            
        return not is_duplicate

    def process_files(self, input_folder, output_folder):
        """处理所有文件"""
        start_time = time.time()
        
        patient_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        unique_records = []
        
        # 处理每个文件
        for filename in patient_files:
            filepath = os.path.join(input_folder, filename)
            patient_id = filename.replace('.txt', '')
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                records = self.parse_medical_record(content, patient_id)
                
                # 为每个记录生成唯一ID并处理
                patient_start_time = time.time()
                patient_unique_records = []
                
                for i, record in enumerate(records):
                    record_id = f"{patient_id}_{i}"
                    is_unique = self.add_record(record_id, record)
                    
                    if is_unique:
                        patient_unique_records.append(record)
                
                # 计算该患者的处理时间
                patient_processing_time = time.time() - patient_start_time
                
                # 计算该患者的性能指标
                self.calculate_patient_metrics(
                    patient_id, records, patient_unique_records, patient_processing_time
                )
                
                # 添加到总唯一记录列表
                unique_records.extend(patient_unique_records)
        
        # 保存去重后的记录
        self.save_deduplicated_records(unique_records, output_folder)
        
        # 计算存储节省
        storage_savings = self.calculate_storage_savings(input_folder, output_folder)
        
        self.processing_time = time.time() - start_time
        
        return {
            "unique_records": len(unique_records),
            "duplicate_count": self.duplicate_count,
            "processing_time": self.processing_time,
            "patient_metrics": self.patient_metrics,
            "statistical_summary": self.get_statistical_summary(),
            **storage_savings
        }


def run_comparison_experiment(input_folder):
    """运行对比实验"""
    # 创建输出文件夹
    output_folders = {
        "mle": "mle_deduplicated",
        "fuzzy": "fuzzy_deduplicated"
    }
    
    for folder in output_folders.values():
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    # 初始化两种去重器
    mle_dedup = MLEDeduplicator(similarity_threshold=0.98)  # 提高MLE阈值
    fuzzy_dedup = FuzzyDeduplicator(threshold=0.4)  # 降低模糊去重阈值
    
    # 运行两种算法
    print("运行MLE确定性去重算法...")
    mle_results = mle_dedup.process_files(input_folder, output_folders["mle"])
    
    print("运行模糊去重算法...")
    fuzzy_results = fuzzy_dedup.process_files(input_folder, output_folders["fuzzy"])
    
    # 汇总结果
    results = {
        "MLE": mle_results,
        "Fuzzy Deduplication": fuzzy_results
    }
    
    return results

def generate_statistical_report(results, output_file="statistical_report.json"):
    """生成统计报告"""
    statistical_report = {}
    
    for method, metrics in results.items():
        statistical_report[method] = metrics["statistical_summary"]
    
    # 保存详细报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(statistical_report, f, indent=2, ensure_ascii=False)
    
    return statistical_report

def generate_comparison_report(results, statistical_report, output_file="comparison_report.json"):
    """生成对比报告"""
    # 创建数据框用于可视化
    df_data = []
    for method, metrics in results.items():
        df_data.append({
            "Method": method,
            "Unique Records": metrics["unique_records"],
            "Duplicate Count": metrics["duplicate_count"],
            "Processing Time (s)": metrics["processing_time"],
            "Original Size (KB)": metrics["original_size_bytes"] / 1024,
            "Deduplicated Size (KB)": metrics["deduplicated_size_bytes"] / 1024,
            "Size Reduction (%)": metrics["size_reduction_percentage"],
            "Compression Ratio": metrics["compression_ratio"]
        })
    
    df = pd.DataFrame(df_data)
    
    # 保存详细报告
    report = {
        "comparison_results": results,
        "statistical_summary": statistical_report,
        "summary": {
            "best_compression_ratio": df.loc[df["Compression Ratio"].idxmax()]["Method"],
            "fastest_processing": df.loc[df["Processing Time (s)"].idxmin()]["Method"],
            "most_duplicates_detected": df.loc[df["Duplicate Count"].idxmax()]["Method"]
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    
    # 打印总结报告
    print("=" * 80)
    print("医疗记录去重算法对比报告")
    print("=" * 80)
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  唯一记录数: {metrics['unique_records']}")
        print(f"  检测到的重复记录数: {metrics['duplicate_count']}")
        print(f"  处理时间: {metrics['processing_time']:.2f} 秒")
        print(f"  原始大小: {metrics['original_size_bytes'] / 1024:.2f} KB")
        print(f"  去重后大小: {metrics['deduplicated_size_bytes'] / 1024:.2f} KB")
        print(f"  存储节省: {metrics['size_reduction_percentage']:.2f}%")
        print(f"  压缩比: {metrics['compression_ratio']:.2f}:1")
    
    print("\n统计摘要 (均值 ± 标准差):")
    for method, stats in statistical_report.items():
        print(f"\n{method}:")
        print(f"  处理时间: {stats.get('processing_time_mean', 0):.4f} ± {stats.get('processing_time_std', 0):.4f} 秒/患者")
        print(f"  存储节省: {stats.get('size_reduction_percentage_mean', 0):.2f} ± {stats.get('size_reduction_percentage_std', 0):.2f} %/患者")
        print(f"  压缩比: {stats.get('compression_ratio_mean', 0):.2f} ± {stats.get('compression_ratio_std', 0):.2f} /患者")
    
    print("\n总结:")
    print(f"  最佳压缩比: {report['summary']['best_compression_ratio']}")
    print(f"  最快处理速度: {report['summary']['fastest_processing']}")
    print(f"  最多重复检测: {report['summary']['most_duplicates_detected']}")
    print("=" * 80)
    
    return df


def main():
    # 设置路径
    input_folder = "patient_reports"
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在!")
        print("请确保已生成患者报告文件并放置在正确的位置。")
        return
    
    # 运行对比实验
    results = run_comparison_experiment(input_folder)
    
    # 生成统计报告
    statistical_report = generate_statistical_report(results)
    
    # 生成对比报告
    df = generate_comparison_report(results, statistical_report)
    
    # 分析结果差异
    analyze_results_discrepancy(results)
    
    
    # 保存详细数据
    df.to_csv("comparison_results.csv", index=False, encoding='utf-8-sig')
    print(f"\n详细对比结果已保存到: comparison_results.csv")
    print(f"完整报告已保存到: comparison_report.json")
    print(f"统计报告已保存到: statistical_report.json")

if __name__ == "__main__":
    main()