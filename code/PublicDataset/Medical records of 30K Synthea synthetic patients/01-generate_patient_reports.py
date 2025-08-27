import pandas as pd
import os
import gc
from datetime import datetime
import time

class MedicalRecordProcessor:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.csv_files = {
            'allergies': 'allergies.csv',
            'careplans': 'careplans.csv',
            'conditions': 'conditions.csv',
            'devices': 'devices.csv',
            'encounters': 'encounters.csv',
            'immunizations': 'immunizations.csv',
            'medications': 'medications.csv',
            'observations': 'observations.csv',
            'organizations': 'organizations.csv',
            'patients': 'patients.csv',
            'payers': 'payers.csv',
            'procedures': 'procedures.csv',
            'providers': 'providers.csv'
        }
        
        # 存储数据索引而不是完整数据
        self.data_indices = {}
        self.providers_dict = {}
        self.organizations_dict = {}
        self.payers_dict = {}
        
        # 进度跟踪
        self.processed_patients = set()
        self.load_checkpoint()
        
    def load_data_indices(self):
        """加载数据索引而不是完整数据"""
        print("正在加载数据索引...")
        
        # 首先加载小型表
        print("加载providers数据...")
        providers_df = pd.read_csv(os.path.join(self.data_folder, self.csv_files['providers']))
        self.providers_dict = providers_df.set_index('Id')['NAME'].to_dict()
        
        print("加载organizations数据...")
        organizations_df = pd.read_csv(os.path.join(self.data_folder, self.csv_files['organizations']))
        self.organizations_dict = organizations_df.set_index('Id')['NAME'].to_dict()
        
        print("加载payers数据...")
        payers_df = pd.read_csv(os.path.join(self.data_folder, self.csv_files['payers']))
        self.payers_dict = payers_df.set_index('Id')['NAME'].to_dict()
        
        # 加载患者列表
        print("加载patients数据...")
        patients_df = pd.read_csv(os.path.join(self.data_folder, self.csv_files['patients']))
        self.patient_ids = patients_df['Id'].tolist()
        
        # 为大型表创建索引
        print("为大型表创建索引...")
        for table_name in ['encounters', 'conditions', 'medications', 'procedures', 
                          'observations', 'immunizations', 'allergies', 'careplans', 'devices']:
            print(f"处理 {table_name} 索引...")
            self.create_index_for_table(table_name)
            
            # 释放内存
            gc.collect()
    
    def create_index_for_table(self, table_name):
        """为表创建索引"""
        file_path = os.path.join(self.data_folder, self.csv_files[table_name])
        
        # 使用迭代器分块处理大文件
        chunk_size = 10000
        index = {}
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            if 'PATIENT' in chunk.columns:
                for patient_id, group in chunk.groupby('PATIENT'):
                    if patient_id not in index:
                        index[patient_id] = []
                    
                    # 只存储必要信息而不是完整数据
                    if table_name == 'encounters':
                        # 对于encounters，我们需要更多信息
                        for _, row in group.iterrows():
                            index[patient_id].append({
                                'Id': row.get('Id', ''),
                                'START': row.get('START', ''),
                                'STOP': row.get('STOP', ''),
                                'ORGANIZATION': row.get('ORGANIZATION', ''),
                                'PROVIDER': row.get('PROVIDER', ''),
                                'PAYER': row.get('PAYER', ''),
                                'ENCOUNTERCLASS': row.get('ENCOUNTERCLASS', ''),
                                'DESCRIPTION': row.get('DESCRIPTION', ''),
                                'BASE_ENCOUNTER_COST': row.get('BASE_ENCOUNTER_COST', 0),
                                'TOTAL_CLAIM_COST': row.get('TOTAL_CLAIM_COST', 0),
                                'REASONCODE': row.get('REASONCODE', ''),
                                'REASONDESCRIPTION': row.get('REASONDESCRIPTION', '')
                            })
                    else:
                        # 对于其他表，存储分块数据
                        index[patient_id].append(chunk[chunk['PATIENT'] == patient_id])
        
        self.data_indices[table_name] = index
    
    def load_checkpoint(self):
        """加载处理进度检查点"""
        checkpoint_file = os.path.join(self.data_folder, 'processed_patients.txt')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                self.processed_patients = set(line.strip() for line in f)
            print(f"已加载检查点，已处理 {len(self.processed_patients)} 个患者")
    
    def save_checkpoint(self, patient_id):
        """保存处理进度检查点"""
        self.processed_patients.add(patient_id)
        checkpoint_file = os.path.join(self.data_folder, 'processed_patients.txt')
        with open(checkpoint_file, 'a') as f:
            f.write(f"{patient_id}\n")
    
    def process_patients(self, output_folder, max_patients=None):
        """处理患者记录"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        print("开始处理患者记录...")
        
        # 过滤已处理的患者
        patients_to_process = [pid for pid in self.patient_ids if pid not in self.processed_patients]
        
        if max_patients:
            patients_to_process = patients_to_process[:max_patients]
        
        total_patients = len(patients_to_process)
        print(f"需要处理 {total_patients} 个患者")
        
        for i, patient_id in enumerate(patients_to_process):
            start_time = time.time()
            print(f"处理患者 {i+1}/{total_patients}: {patient_id}")
            
            try:
                self.process_single_patient(patient_id, output_folder)
                self.save_checkpoint(patient_id)
                
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"患者 {patient_id} 处理完成，耗时: {elapsed:.2f} 秒")
                
                # 估计剩余时间
                remaining = total_patients - (i + 1)
                if remaining > 0 and i > 0:
                    avg_time = (time.time() - start_time) / (i + 1)
                    eta = avg_time * remaining
                    print(f"预计剩余时间: {eta/60:.2f} 分钟")
                
            except Exception as e:
                print(f"处理患者 {patient_id} 时出错: {str(e)}")
            
            # 定期释放内存
            if i % 100 == 0:
                gc.collect()
    
    def process_single_patient(self, patient_id, output_folder):
        """处理单个患者的记录"""
        # 获取患者基本信息
        patients_file = os.path.join(self.data_folder, self.csv_files['patients'])
        patient_info = None
        
        for chunk in pd.read_csv(patients_file, chunksize=1000, low_memory=False):
            patient_data = chunk[chunk['Id'] == patient_id]
            if not patient_data.empty:
                patient_info = patient_data.iloc[0]
                break
        
        if patient_info is None:
            print(f"未找到患者 {patient_id} 的信息")
            return
        
        # 获取患者的所有就诊记录
        encounters = self.data_indices['encounters'].get(patient_id, [])
        
        # 如果没有找到就诊记录，尝试直接从文件查找
        if not encounters:
            encounters_file = os.path.join(self.data_folder, self.csv_files['encounters'])
            for chunk in pd.read_csv(encounters_file, chunksize=10000, low_memory=False):
                patient_encounters = chunk[chunk['PATIENT'] == patient_id]
                if not patient_encounters.empty:
                    encounters = patient_encounters.to_dict('records')
                    break
        
        # 按时间排序就诊记录
        try:
            encounters.sort(key=lambda x: pd.to_datetime(x['START']))
        except:
            # 如果日期解析失败，保持原顺序
            pass
        
        # 为每个就诊记录收集相关信息
        patient_encounters_data = []
        
        for encounter in encounters:
            encounter_id = encounter.get('Id', '')
            encounter_date = encounter.get('START', '')
            
            # 收集本次就诊的所有相关信息
            encounter_data = {
                'date': encounter_date,
                'encounter_id': encounter_id,
                'encounter_type': encounter.get('ENCOUNTERCLASS', ''),
                'description': encounter.get('DESCRIPTION', ''),
                'reason': encounter.get('REASONDESCRIPTION', ''),
                'cost': encounter.get('TOTAL_CLAIM_COST', 0),
                'provider': self.providers_dict.get(encounter.get('PROVIDER', ''), '未知提供者'),
                'organization': self.organizations_dict.get(encounter.get('ORGANIZATION', ''), '未知组织'),
                'payer': self.payers_dict.get(encounter.get('PAYER', ''), '未知支付方'),
                'conditions': self.get_patient_data('conditions', patient_id, encounter_id),
                'medications': self.get_patient_data('medications', patient_id, encounter_id),
                'procedures': self.get_patient_data('procedures', patient_id, encounter_id),
                'observations': self.get_patient_data('observations', patient_id, encounter_id),
                'immunizations': self.get_patient_data('immunizations', patient_id, encounter_id),
                'allergies': self.get_patient_data('allergies', patient_id, encounter_id),
                'careplans': self.get_patient_data('careplans', patient_id, encounter_id),
                'devices': self.get_patient_data('devices', patient_id, encounter_id)
            }
            
            patient_encounters_data.append(encounter_data)
        
        # 生成患者报告
        self.generate_patient_report(patient_id, patient_info, patient_encounters_data, output_folder)
    
    def get_patient_data(self, table_name, patient_id, encounter_id):
        """获取患者的特定数据"""
        result = []
        
        # 首先从索引中查找
        if table_name in self.data_indices and patient_id in self.data_indices[table_name]:
            for data_chunk in self.data_indices[table_name][patient_id]:
                if isinstance(data_chunk, pd.DataFrame):
                    # 如果是DataFrame，筛选出匹配的记录
                    filtered = data_chunk[data_chunk['ENCOUNTER'] == encounter_id]
                    if not filtered.empty:
                        result.extend(filtered.to_dict('records'))
                else:
                    # 如果是字典列表，直接筛选
                    if data_chunk.get('ENCOUNTER') == encounter_id:
                        result.append(data_chunk)
        
        # 如果索引中没有找到，直接从文件查找
        if not result:
            file_path = os.path.join(self.data_folder, self.csv_files[table_name])
            for chunk in pd.read_csv(file_path, chunksize=10000, low_memory=False):
                filtered = chunk[(chunk['PATIENT'] == patient_id) & (chunk['ENCOUNTER'] == encounter_id)]
                if not filtered.empty:
                    result.extend(filtered.to_dict('records'))
                # 如果已经找到一些数据，可以提前退出
                if result and len(result) > 10:  # 假设每个就诊记录不会有太多相关数据
                    break
        
        return result
    
    def generate_patient_report(self, patient_id, patient_info, encounters, output_folder):
        """生成单个患者的报告"""
        # 获取患者姓名
        first_name = patient_info.get('FIRST', '')
        last_name = patient_info.get('LAST', '')
        patient_name = f"{first_name}_{last_name}" if first_name and last_name else patient_id
        
        # 创建报告文件
        filename = f"patient_{patient_name}_{patient_id[:8]}.txt"
        filepath = os.path.join(output_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # 写入患者基本信息
            f.write("=" * 60 + "\n")
            f.write(f"患者医疗记录报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"患者ID: {patient_id}\n")
            f.write(f"姓名: {patient_info.get('FIRST', '')} {patient_info.get('LAST', '')}\n")
            f.write(f"性别: {patient_info.get('GENDER', '')}\n")
            f.write(f"出生日期: {patient_info.get('BIRTHDATE', '')}\n")
            if pd.notna(patient_info.get('DEATHDATE')):
                f.write(f"死亡日期: {patient_info.get('DEATHDATE', '')}\n")
            f.write(f"地址: {patient_info.get('ADDRESS', '')}, {patient_info.get('CITY', '')}, {patient_info.get('STATE', '')}\n")
            f.write("\n")
            
            # 写入每次就诊记录
            for i, encounter in enumerate(encounters):
                f.write(f"就诊 #{i+1}: {encounter['date']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"就诊类型: {encounter['encounter_type']}\n")
                f.write(f"描述: {encounter['description']}\n")
                if encounter['reason']:
                    f.write(f"原因: {encounter['reason']}\n")
                f.write(f"费用: ${encounter['cost']:.2f}\n")
                f.write(f"提供者: {encounter['provider']}\n")
                f.write(f"机构: {encounter['organization']}\n")
                f.write(f"支付方: {encounter['payer']}\n")
                
                # 写入诊断信息
                if encounter['conditions']:
                    f.write("\n诊断:\n")
                    for condition in encounter['conditions']:
                        start = condition.get('START', '')
                        stop = condition.get('STOP', '')
                        desc = condition.get('DESCRIPTION', '')
                        duration = f" ({start} - {stop})" if stop and pd.notna(stop) else f" (从 {start} 开始)"
                        f.write(f"  - {desc}{duration}\n")
                
                # 写入药物信息
                if encounter['medications']:
                    f.write("\n药物:\n")
                    for med in encounter['medications']:
                        start = med.get('START', '')
                        stop = med.get('STOP', '')
                        desc = med.get('DESCRIPTION', '')
                        cost = med.get('TOTALCOST', 0) or med.get('BASE_COST', 0)
                        duration = f" ({start} - {stop})" if stop and pd.notna(stop) else f" (从 {start} 开始)"
                        f.write(f"  - {desc}{duration} - 费用: ${cost:.2f}\n")
                
                # 写入程序/手术信息
                if encounter['procedures']:
                    f.write("\n程序/手术:\n")
                    for proc in encounter['procedures']:
                        start = proc.get('START', '')
                        stop = proc.get('STOP', '')
                        desc = proc.get('DESCRIPTION', '')
                        cost = proc.get('BASE_COST', 0)
                        duration = f" ({start} - {stop})" if stop and pd.notna(stop) and start != stop else f" ({start})"
                        f.write(f"  - {desc}{duration} - 费用: ${cost:.2f}\n")
                
                # 写入观察结果
                if encounter['observations']:
                    f.write("\n观察结果:\n")
                    for obs in encounter['observations']:
                        date = obs.get('DATE', '')
                        desc = obs.get('DESCRIPTION', '')
                        value = obs.get('VALUE', '')
                        units = obs.get('UNITS', '')
                        f.write(f"  - {desc}: {value} {units} ({date})\n")
                
                # 写入其他信息
                for category in ['immunizations', 'allergies', 'careplans', 'devices']:
                    if encounter[category]:
                        category_name = {
                            'immunizations': '免疫',
                            'allergies': '过敏',
                            'careplans': '护理计划',
                            'devices': '医疗设备'
                        }[category]
                        
                        f.write(f"\n{category_name}:\n")
                        for item in encounter[category]:
                            if category == 'immunizations':
                                f.write(f"  - {item.get('DESCRIPTION', '')} - 费用: ${item.get('BASE_COST', 0):.2f} ({item.get('DATE', '')})\n")
                            elif category == 'allergies':
                                f.write(f"  - {item.get('DESCRIPTION', '')} - 反应: {item.get('REACTION1', '')} ({item.get('SEVERITY1', '')})\n")
                            elif category == 'careplans':
                                f.write(f"  - {item.get('DESCRIPTION', '')} - 原因: {item.get('REASONDESCRIPTION', '')}\n")
                            elif category == 'devices':
                                f.write(f"  - {item.get('DESCRIPTION', '')}\n")
                
                f.write("\n" + "=" * 60 + "\n\n")

# 使用示例
if __name__ == "__main__":
    # 设置CSV文件所在文件夹路径
    csv_folder = "csv"  # 替换为您的CSV文件夹路径
    
    # 设置输出文件夹路径
    output_folder = "patient_reports"
    
    # 创建处理器实例
    processor = MedicalRecordProcessor(csv_folder)
    
    # 加载数据索引
    processor.load_data_indices()
    
    # 处理患者记录（可以设置最大处理数量进行测试）
    processor.process_patients(output_folder, max_patients=100)  # 先处理100个患者进行测试
    
    print("患者报告生成完成!")