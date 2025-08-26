import pandas as pd
import os
import glob
from datetime import datetime
import json

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
        
        # 加载数据
        self.data = {}
        self.load_data()
        
        # 处理后的患者记录
        self.patient_records = {}
        
    def load_data(self):
        """加载所有CSV文件到内存"""
        print("正在加载数据...")
        for key, filename in self.csv_files.items():
            try:
                filepath = os.path.join(self.data_folder, filename)
                self.data[key] = pd.read_csv(filepath, low_memory=False)
                print(f"已加载: {filename} (行数: {len(self.data[key])})")
            except Exception as e:
                print(f"加载 {filename} 时出错: {str(e)}")
    
    def process_patient_records(self):
        """处理所有患者的记录"""
        print("正在处理患者记录...")
        
        # 获取所有患者ID
        patient_ids = self.data['patients']['Id'].unique()
        
        for patient_id in patient_ids:
            self.process_single_patient(patient_id)
    
    def process_single_patient(self, patient_id):
        """处理单个患者的记录"""
        # 获取患者基本信息
        patient_info = self.data['patients'][self.data['patients']['Id'] == patient_id].iloc[0]
        
        # 获取患者的所有就诊记录
        encounters = self.data['encounters'][self.data['encounters']['PATIENT'] == patient_id]
        
        # 按时间排序就诊记录（从最早到最晚）
        encounters['START'] = pd.to_datetime(encounters['START'])
        encounters = encounters.sort_values('START')
        
        # 为每个就诊记录收集相关信息
        patient_encounters = []
        
        for _, encounter in encounters.iterrows():
            encounter_id = encounter['Id']
            encounter_date = encounter['START']
            
            # 收集本次就诊的所有相关信息
            encounter_data = {
                'date': encounter_date,
                'encounter_id': encounter_id,
                'encounter_type': encounter['ENCOUNTERCLASS'],
                'description': encounter['DESCRIPTION'],
                'reason': encounter.get('REASONDESCRIPTION', ''),
                'cost': encounter['TOTAL_CLAIM_COST'],
                'provider': self.get_provider_info(encounter['PROVIDER']),
                'organization': self.get_organization_info(encounter['ORGANIZATION']),
                'payer': self.get_payer_info(encounter['PAYER']),
                'conditions': self.get_conditions(patient_id, encounter_id),
                'medications': self.get_medications(patient_id, encounter_id),
                'procedures': self.get_procedures(patient_id, encounter_id),
                'observations': self.get_observations(patient_id, encounter_id),
                'immunizations': self.get_immunizations(patient_id, encounter_id),
                'allergies': self.get_allergies(patient_id, encounter_id),
                'careplans': self.get_careplans(patient_id, encounter_id),
                'devices': self.get_devices(patient_id, encounter_id)
            }
            
            patient_encounters.append(encounter_data)
        
        # 存储患者记录（按时间升序排列，最早的在最前）
        self.patient_records[patient_id] = {
            'info': patient_info,
            'encounters': patient_encounters
        }
    
    def get_provider_info(self, provider_id):
        """获取提供者信息"""
        provider = self.data['providers'][self.data['providers']['Id'] == provider_id]
        if not provider.empty:
            return provider.iloc[0]['NAME']
        return "未知提供者"
    
    def get_organization_info(self, org_id):
        """获取组织信息"""
        org = self.data['organizations'][self.data['organizations']['Id'] == org_id]
        if not org.empty:
            return org.iloc[0]['NAME']
        return "未知组织"
    
    def get_payer_info(self, payer_id):
        """获取支付方信息"""
        payer = self.data['payers'][self.data['payers']['Id'] == payer_id]
        if not payer.empty:
            return payer.iloc[0]['NAME']
        return "未知支付方"
    
    def get_conditions(self, patient_id, encounter_id):
        """获取诊断条件"""
        conditions = self.data['conditions'][
            (self.data['conditions']['PATIENT'] == patient_id) & 
            (self.data['conditions']['ENCOUNTER'] == encounter_id)
        ]
        return conditions[['START', 'STOP', 'DESCRIPTION']].to_dict('records')
    
    def get_medications(self, patient_id, encounter_id):
        """获取药物信息"""
        meds = self.data['medications'][
            (self.data['medications']['PATIENT'] == patient_id) & 
            (self.data['medications']['ENCOUNTER'] == encounter_id)
        ]
        return meds[['START', 'STOP', 'DESCRIPTION', 'TOTALCOST']].to_dict('records')
    
    def get_procedures(self, patient_id, encounter_id):
        """获取程序/手术信息"""
        procedures = self.data['procedures'][
            (self.data['procedures']['PATIENT'] == patient_id) & 
            (self.data['procedures']['ENCOUNTER'] == encounter_id)
        ]
        return procedures[['START', 'STOP', 'DESCRIPTION', 'BASE_COST']].to_dict('records')
    
    def get_observations(self, patient_id, encounter_id):
        """获取观察结果"""
        observations = self.data['observations'][
            (self.data['observations']['PATIENT'] == patient_id) & 
            (self.data['observations']['ENCOUNTER'] == encounter_id)
        ]
        return observations[['DATE', 'DESCRIPTION', 'VALUE', 'UNITS']].to_dict('records')
    
    def get_immunizations(self, patient_id, encounter_id):
        """获取免疫信息"""
        immunizations = self.data['immunizations'][
            (self.data['immunizations']['PATIENT'] == patient_id) & 
            (self.data['immunizations']['ENCOUNTER'] == encounter_id)
        ]
        return immunizations[['DATE', 'DESCRIPTION', 'BASE_COST']].to_dict('records')
    
    def get_allergies(self, patient_id, encounter_id):
        """获取过敏信息"""
        allergies = self.data['allergies'][
            (self.data['allergies']['PATIENT'] == patient_id) & 
            (self.data['allergies']['ENCOUNTER'] == encounter_id)
        ]
        return allergies[['START', 'DESCRIPTION', 'REACTION1', 'SEVERITY1']].to_dict('records')
    
    def get_careplans(self, patient_id, encounter_id):
        """获取护理计划"""
        careplans = self.data['careplans'][
            (self.data['careplans']['PATIENT'] == patient_id) & 
            (self.data['careplans']['ENCOUNTER'] == encounter_id)
        ]
        return careplans[['START', 'STOP', 'DESCRIPTION', 'REASONDESCRIPTION']].to_dict('records')
    
    def get_devices(self, patient_id, encounter_id):
        """获取设备信息"""
        devices = self.data['devices'][
            (self.data['devices']['PATIENT'] == patient_id) & 
            (self.data['devices']['ENCOUNTER'] == encounter_id)
        ]
        return devices[['START', 'STOP', 'DESCRIPTION']].to_dict('records')
    
    def generate_patient_reports(self, output_folder):
        """为每位患者生成报告"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        print(f"正在生成患者报告到 {output_folder}...")
        
        for patient_id, record in self.patient_records.items():
            # 获取患者姓名
            first_name = record['info']['FIRST']
            last_name = record['info']['LAST']
            patient_name = f"{first_name}_{last_name}" if pd.notna(first_name) and pd.notna(last_name) else patient_id
            
            # 创建报告文件
            filename = f"patient_{patient_name}_{patient_id[:8]}.txt"
            filepath = os.path.join(output_folder, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入患者基本信息
                f.write("=" * 60 + "\n")
                f.write(f"患者医疗记录报告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"患者ID: {patient_id}\n")
                f.write(f"姓名: {record['info'].get('FIRST', '')} {record['info'].get('LAST', '')}\n")
                f.write(f"性别: {record['info'].get('GENDER', '')}\n")
                f.write(f"出生日期: {record['info'].get('BIRTHDATE', '')}\n")
                if pd.notna(record['info'].get('DEATHDATE')):
                    f.write(f"死亡日期: {record['info'].get('DEATHDATE', '')}\n")
                f.write(f"地址: {record['info'].get('ADDRESS', '')}, {record['info'].get('CITY', '')}, {record['info'].get('STATE', '')}\n")
                f.write("\n")
                
                # 反转就诊记录顺序，使最新的记录在前
                encounters_reversed = list(reversed(record['encounters']))
                
                # 写入每次就诊记录（从最新到最旧）
                for i, encounter in enumerate(encounters_reversed):
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
                            duration = f" ({start} - {stop})" if pd.notna(stop) else f" (从 {start} 开始)"
                            f.write(f"  - {desc}{duration}\n")
                    
                    # 写入药物信息
                    if encounter['medications']:
                        f.write("\n药物:\n")
                        for med in encounter['medications']:
                            start = med.get('START', '')
                            stop = med.get('STOP', '')
                            desc = med.get('DESCRIPTION', '')
                            cost = med.get('TOTALCOST', 0)
                            duration = f" ({start} - {stop})" if pd.notna(stop) else f" (从 {start} 开始)"
                            f.write(f"  - {desc}{duration} - 费用: ${cost:.2f}\n")
                    
                    # 写入程序/手术信息
                    if encounter['procedures']:
                        f.write("\n程序/手术:\n")
                        for proc in encounter['procedures']:
                            start = proc.get('START', '')
                            stop = proc.get('STOP', '')
                            desc = proc.get('DESCRIPTION', '')
                            cost = proc.get('BASE_COST', 0)
                            duration = f" ({start} - {stop})" if pd.notna(stop) and start != stop else f" ({start})"
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
            
            print(f"已生成报告: {filename}")
        
        print(f"已完成所有患者报告生成!")

# 使用示例
if __name__ == "__main__":
    # 设置CSV文件所在文件夹路径
    csv_folder = "csv"  # 替换为您的CSV文件夹路径
    
    # 设置输出文件夹路径
    output_folder = "patient_reports"
    
    # 创建处理器实例
    processor = MedicalRecordProcessor(csv_folder)
    
    # 处理患者记录
    processor.process_patient_records()
    
    # 生成患者报告
    processor.generate_patient_reports(output_folder)