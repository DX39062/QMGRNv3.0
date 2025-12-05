import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from data_loader import load_adjacency_matrix, MultimodalDataset
from utils import setup_logger
from models import sGCN_LSTM, sGCN_QLSTM, sGCN_Only

def get_args():
    parser = argparse.ArgumentParser(description="sGCN-LSTM/QLSTM Training Script")
    
    # 模型选择
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'qlstm', 'gcn_only'], 
                        help='Choose model architecture: lstm, qlstm, or gcn_only')
    
    # 超参数
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds')
    
    # 结构门控偏置 (Bias) - 默认值为 0.1
    parser.add_argument('--struct_bias', type=float, default=0.1, 
                        help='Bias value added to the structure gate mask (default: 0.1)')
    
    # 路径配置
    parser.add_argument('--data_path', type=str, default='./datasets', help='Root path for datasets')
    parser.add_argument('--device', type=str, default='auto', help='Device: "cuda", "cpu", or "auto"')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. 设备配置
    if args.device == 'auto':
        if args.model == 'qlstm':
            device = torch.device('cpu')
            print("Note: Defaulting to CPU for QLSTM to avoid simulator mismatch.")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    # 2. 构建日志文件名后缀 (仅当参数非默认时添加)
    extra_suffix = ""
    
    # 检查 struct_bias 是否为默认值 0.1
    # 如果用户指定了 --struct_bias 0.2，这里就会触发，文件名会带上 _bias0.2
    if args.struct_bias != 0.1:
        extra_suffix += f"_bias{args.struct_bias}"
        
    # 可以在这里添加更多非默认参数的检查
    # 例如: if args.epochs != 80: extra_suffix += f"_ep{args.epochs}"

    # 3. 初始化日志
    log_file = setup_logger(args.model, extra_info=extra_suffix)
    
    logging.info(f"=== Training Configuration ===")
    logging.info(f" Model: {args.model}")
    logging.info(f" Bias: {args.struct_bias}")
    logging.info(f" Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    logging.info(f" Device: {device}")
    logging.info(f" Log File: {log_file}")
    logging.info(f"==============================")

    # 4. 数据路径
    fmri_dir = os.path.join(args.data_path, "fMRI")
    smri_file = os.path.join(args.data_path, "GMV_Node_Features.csv")
    label_file = os.path.join(args.data_path, "labels.csv")
    adj_file = os.path.join(args.data_path, "FC.csv")

    if not os.path.exists(adj_file):
        logging.error(f"Missing adjacency file: {adj_file}")
        return

    # 5. 加载数据
    adj_static = load_adjacency_matrix(adj_file).to(device)
    full_dataset = MultimodalDataset(fmri_dir, smri_file, label_file)
    all_labels = np.array(full_dataset.labels_list)
    all_indices = np.arange(len(full_dataset))

    # 6. K-Fold 循环
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results = []
    
    logging.info(f"Starting {args.folds}-Fold Cross Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
        logging.info(f"\n=== Fold {fold+1}/{args.folds} ===")
        fold_start_time = time.time()
        
        # 数据集切分
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        # 加权采样
        y_train = all_labels[train_idx]
        class_counts = np.bincount(y_train)
        class_weights = np.nan_to_num(1. / class_counts, posinf=0.0)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).float(), 
                                        num_samples=len(train_idx), replacement=True)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        
        # 模型初始化
        if args.model == 'lstm':
            model = sGCN_LSTM(struct_bias=args.struct_bias).to(device)
        elif args.model == 'qlstm':
            model = sGCN_QLSTM(struct_bias=args.struct_bias).to(device)
        elif args.model == 'gcn_only':
            model = sGCN_Only(struct_bias=args.struct_bias).to(device)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        
        best_fold_bacc = 0.0
        best_fold_acc = 0.0
        
        # Epoch 循环
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            
            for fmri, smri, labels in train_loader:
                fmri, smri, labels = fmri.to(device), smri.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(fmri, smri, adj_static)
                loss = criterion(outputs, labels)
                loss.backward()
                
                if args.model != 'qlstm': 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                running_loss += loss.item()

            # 验证
            model.eval()
            preds_list = []
            targets_list = []
            
            with torch.no_grad():
                for fmri, smri, labels in val_loader:
                    fmri, smri, labels = fmri.to(device), smri.to(device), labels.to(device)
                    outputs = model(fmri, smri, adj_static)
                    _, predicted = torch.max(outputs.data, 1)
                    preds_list.extend(predicted.cpu().numpy())
                    targets_list.extend(labels.cpu().numpy())
            
            epoch_acc = np.mean(np.array(preds_list) == np.array(targets_list)) * 100
            epoch_bacc = balanced_accuracy_score(targets_list, preds_list) * 100
            
            if epoch_bacc > best_fold_bacc:
                best_fold_bacc = epoch_bacc
                best_fold_acc = epoch_acc
            
            if (epoch+1) % 10 == 0:
                logging.info(f"  [Fold {fold+1}] Epoch {epoch+1}: Loss {running_loss/len(train_loader):.4f} | Val B-Acc: {epoch_bacc:.2f}%")

        logging.info(f" Fold {fold+1} Finished. Best B-Acc: {best_fold_bacc:.2f}%")
        fold_results.append({'fold': fold+1, 'bacc': best_fold_bacc, 'acc': best_fold_acc})

    # 汇总
    logging.info("\n" + "="*30)
    logging.info(f"   {args.model.upper()} CV Results   ")
    logging.info("="*30)
    avg_bacc = np.mean([r['bacc'] for r in fold_results])
    avg_acc = np.mean([r['acc'] for r in fold_results])
    
    for res in fold_results:
        logging.info(f"Fold {res['fold']}: B-Acc = {res['bacc']:.2f}% | Acc = {res['acc']:.2f}%")
        
    logging.info("-" * 30)
    logging.info(f"Average B-Acc: {avg_bacc:.2f}%")
    logging.info(f"Average Acc  : {avg_acc:.2f}%")
    print(f"Done. Logs saved to {log_file}")

if __name__ == "__main__":
    main()