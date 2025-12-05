import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import logging

def load_adjacency_matrix(adj_file, n_nodes=116):
    logging.info(f"--- 正在加载邻接矩阵: {adj_file} ---")
    try:
        adj_np = pd.read_csv(adj_file, header=None).values.astype(np.float32)
        if adj_np.shape != (n_nodes, n_nodes):
            raise ValueError(f"邻接矩阵形状错误: 期望 ({n_nodes}, {n_nodes}), 实际 {adj_np.shape}")
        
        np.fill_diagonal(adj_np, 0)
        adj_torch = torch.tensor(adj_np)
        
        A_tilde = adj_torch + torch.eye(n_nodes)
        degrees = torch.sum(A_tilde, dim=1)
        D_inv_sqrt = torch.pow(degrees, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        D_mat = torch.diag(D_inv_sqrt)
        adj_normalized = D_mat @ A_tilde @ D_mat
        
        logging.info(" 邻接矩阵加载并归一化完成。")
        return adj_normalized
    except Exception as e:
        logging.error(f"加载邻接矩阵时出错: {e}")
        raise e

class MultimodalDataset(Dataset):
    def __init__(self, fmri_dir, smri_file, label_file, n_time_steps=140, n_nodes=116):
        self.fmri_dir = fmri_dir
        self.n_time_steps = n_time_steps
        self.n_nodes = n_nodes
        
        # 1. 解析标签
        self.label_map = {}
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if ':' in line:
                        parts = line.strip().split(':')
                        clean_id = parts[0].strip().replace("'", "").replace('"', "")
                        clean_label = int(parts[1].strip().replace(",", ""))
                        self.label_map[clean_id] = clean_label
        except Exception as e:
            logging.error(f"解析标签文件失败: {e}")
            raise e
            
        # 2. 加载 sMRI 并清洗
        if not os.path.exists(smri_file):
            raise FileNotFoundError(f"找不到 sMRI 文件: {smri_file}")
            
        logging.info(f"--- 正在加载并清洗 sMRI 数据: {smri_file} ---")
        self.smri_df = pd.read_csv(smri_file)
        
        if 'Subject_ID' in self.smri_df.columns:
            self.smri_df = self.smri_df.set_index('Subject_ID')
        else:
            self.smri_df = self.smri_df.set_index(self.smri_df.columns[0])

        # 数据清洗: 剔除含 0 的样本
        initial_count = len(self.smri_df)
        valid_mask = (self.smri_df != 0).all(axis=1)
        self.smri_df = self.smri_df[valid_mask]
        
        dropped_count = initial_count - len(self.smri_df)
        if dropped_count > 0:
            logging.warning(f" 警告: 已剔除 {dropped_count} 个含有 0 值(异常脑区)的样本。")
        else:
            logging.info(" sMRI 数据质量良好，未发现含 0 值的样本。")

        # 3. 匹配数据
        self.data_list = [] 
        self.labels_list = [] 
        
        if not os.path.exists(fmri_dir):
            raise FileNotFoundError(f"找不到 fMRI 文件夹: {fmri_dir}")
            
        fmri_files = sorted([f for f in os.listdir(fmri_dir) if f.endswith('.csv')])
        
        match_count = 0
        for f_file in fmri_files:
            long_id = os.path.splitext(f_file)[0]
            parts = long_id.split('_')
            if len(parts) >= 3:
                short_id = "_".join(parts[:3])
            else:
                short_id = long_id
            
            if short_id in self.label_map and long_id in self.smri_df.index:
                label = self.label_map[short_id]
                f_path = os.path.join(fmri_dir, f_file)
                self.data_list.append((f_path, long_id, label))
                self.labels_list.append(label)
                match_count += 1
                
        logging.info(f" 最终匹配完成: 共有 {match_count} 个有效样本。")
        if match_count == 0:
            raise RuntimeError("数据匹配失败。")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        f_path, subject_id, label = self.data_list[idx]
        try:
            # 读取 fMRI
            fmri_data = pd.read_csv(f_path, header=None).values.astype(np.float32)
            if fmri_data.shape != (self.n_time_steps, self.n_nodes):
                if fmri_data.shape == (self.n_nodes, self.n_time_steps):
                    fmri_data = fmri_data.T
                elif fmri_data.shape[0] > self.n_time_steps:
                    fmri_data = fmri_data[:self.n_time_steps, :]
                else:
                    padding = np.zeros((self.n_time_steps - fmri_data.shape[0], self.n_nodes))
                    fmri_data = np.vstack([fmri_data, padding])

            scaler = StandardScaler()
            fmri_data = scaler.fit_transform(fmri_data) 

            # 读取 sMRI
            smri_row = self.smri_df.loc[subject_id].values.astype(np.float32)
            if smri_row.shape[0] != self.n_nodes:
                 smri_row = smri_row[:self.n_nodes]
            
            # 归一化
            if np.std(smri_row) > 0:
                smri_row = (smri_row - np.mean(smri_row)) / np.std(smri_row)
            else:
                smri_row = smri_row - np.mean(smri_row)

            return (
                torch.tensor(fmri_data, dtype=torch.float32), 
                torch.tensor(smri_row, dtype=torch.float32), 
                torch.tensor(label, dtype=torch.long)
            )
        except Exception as e:
            logging.error(f"读取数据出错 (ID: {subject_id}): {e}")
            return (torch.zeros(self.n_time_steps, self.n_nodes), 
                    torch.zeros(self.n_nodes), 
                    torch.tensor(0, dtype=torch.long))