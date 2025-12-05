import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# 尝试导入 pennylane，如果失败则标记不可用，只有在调用 QLSTM 时才报错
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

# ==========================================
# 基础组件: StructureGatedGCN (带 Bias 参数)
# ==========================================
class StructureGatedGCN(nn.Module):
    def __init__(self, n_nodes=116, feature_dim=64, struct_bias=0.1):
        super(StructureGatedGCN, self).__init__()
        self.n_nodes = n_nodes
        self.struct_bias = struct_bias  # 可配置的 Bias
        
        self.struct_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.fmri_linear = nn.Linear(1, feature_dim)
        self.residual_proj = nn.Linear(1, feature_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, fmri, smri_gmv, adj_static):
        # 结构门控 (Bias 使用 self.struct_bias)
        node_integrity = self.struct_gate(smri_gmv.unsqueeze(-1)) 
        struct_mask = (node_integrity @ node_integrity.transpose(1, 2)) + self.struct_bias
        adj_dynamic = adj_static.unsqueeze(0) * struct_mask
        
        # GCN
        fmri_feat = self.fmri_linear(fmri.unsqueeze(-1)) 
        fmri_feat = F.relu(fmri_feat)
        
        out = torch.einsum('bmn, btnf -> btmf', adj_dynamic, fmri_feat)
        
        # Residual
        residual = self.residual_proj(fmri.unsqueeze(-1))
        out = out + residual 
        
        out = F.relu(out)
        out = self.dropout(out)
        out = torch.mean(out, dim=2) 
        return out 

# ==========================================
# 辅助组件: QLSTM Cell
# ==========================================
if HAS_PENNYLANE:
    n_qubits = 8
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def quantum_gate_circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[:, i], wires=i)
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class QLSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=8, n_qlayers=1):
        super(QLSTM_Cell, self).__init__()
        if not HAS_PENNYLANE:
            raise ImportError("Pennylane not installed, cannot use QLSTM.")
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        
        self.cl_input_map = nn.Linear(input_size + hidden_size, n_qubits)
        
        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        self.vqc_forget = qml.qnn.TorchLayer(quantum_gate_circuit, weight_shapes)
        self.vqc_input = qml.qnn.TorchLayer(quantum_gate_circuit, weight_shapes)
        self.vqc_update = qml.qnn.TorchLayer(quantum_gate_circuit, weight_shapes)
        self.vqc_output = qml.qnn.TorchLayer(quantum_gate_circuit, weight_shapes)

    def forward(self, x, init_states=None):
        B, T, _ = x.size()
        if init_states is None:
            h_t = torch.zeros(B, self.hidden_size).to(x.device)
            c_t = torch.zeros(B, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states
            
        hidden_seq = []
        for t in range(T):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, h_t), dim=1)
            q_in = torch.atan(self.cl_input_map(combined)) 
            
            f_t = torch.sigmoid(self.vqc_forget(q_in))
            i_t = torch.sigmoid(self.vqc_input(q_in))
            g_t = torch.tanh(self.vqc_update(q_in)) 
            o_t = torch.sigmoid(self.vqc_output(q_in))
            
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))
            
        hidden_seq = torch.cat(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)

# ==========================================
# 模型 1: sGCN-LSTM (Classic)
# ==========================================
class sGCN_LSTM(nn.Module):
    def __init__(self, n_nodes=116, n_classes=2, struct_bias=0.1):
        super(sGCN_LSTM, self).__init__()
        self.hidden_dim = 64  
        self.pool_kernel = 4 
        
        self.struct_gcn = StructureGatedGCN(n_nodes=n_nodes, feature_dim=self.hidden_dim, struct_bias=struct_bias)
        
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=1, 
            batch_first=True,
            dropout=0.0
        )
        
        self.bn_final = nn.BatchNorm1d(self.hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, fmri, smri, adj_static):
        gcn_out = self.struct_gcn(fmri, smri, adj_static)
        gcn_out = gcn_out.permute(0, 2, 1) 
        gcn_out = F.avg_pool1d(gcn_out, kernel_size=self.pool_kernel)
        lstm_in = gcn_out.permute(0, 2, 1)
        
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(lstm_in)
        
        feat = self.bn_final(h_n[-1])
        return self.classifier(feat)

# ==========================================
# 模型 2: sGCN-QLSTM (Quantum)
# ==========================================
class sGCN_QLSTM(nn.Module):
    def __init__(self, n_nodes=116, n_classes=2, struct_bias=0.1):
        super(sGCN_QLSTM, self).__init__()
        if not HAS_PENNYLANE:
            raise RuntimeError("Cannot initialize sGCN_QLSTM: Pennylane library missing.")
            
        self.gcn_dim = 64  
        self.n_qubits = 8
        self.pool_kernel = 4 
        
        self.struct_gcn = StructureGatedGCN(n_nodes=n_nodes, feature_dim=self.gcn_dim, struct_bias=struct_bias)
        self.bridge = nn.Linear(self.gcn_dim, self.n_qubits)
        self.qlstm = QLSTM_Cell(input_size=self.n_qubits, hidden_size=self.n_qubits, n_qubits=self.n_qubits)
        self.bn_final = nn.BatchNorm1d(self.n_qubits)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, fmri, smri, adj_static):
        gcn_out = self.struct_gcn(fmri, smri, adj_static)
        gcn_out = gcn_out.permute(0, 2, 1) 
        gcn_out = F.avg_pool1d(gcn_out, kernel_size=self.pool_kernel)
        gcn_out = gcn_out.permute(0, 2, 1)
        
        qlstm_in = self.bridge(gcn_out)
        _, (h_n, _) = self.qlstm(qlstm_in)
        
        feat = self.bn_final(h_n)
        return self.classifier(feat)

# ==========================================
# 模型 3: sGCN-Only (No LSTM)
# ==========================================
class sGCN_Only(nn.Module):
    def __init__(self, n_nodes=116, n_classes=2, struct_bias=0.1):
        super(sGCN_Only, self).__init__()
        self.hidden_dim = 64  
        
        self.struct_gcn = StructureGatedGCN(n_nodes=n_nodes, feature_dim=self.hidden_dim, struct_bias=struct_bias)
        self.bn_final = nn.BatchNorm1d(self.hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, fmri, smri, adj_static):
        gcn_out = self.struct_gcn(fmri, smri, adj_static)
        # 全局时间平均池化
        feat = torch.mean(gcn_out, dim=1)
        
        feat = self.bn_final(feat)
        return self.classifier(feat)