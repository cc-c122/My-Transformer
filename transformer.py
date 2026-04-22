import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------------
# 设置随机种子
# --------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# --------------------------
# 0. Mask 生成工具函数
# --------------------------
def generate_padding_mask(seq, pad_idx=0):
    """
    生成 Padding Mask
    Args:
        seq: [batch_size, seq_len]
        pad_idx: padding token 的索引
    Returns:
        mask: [batch_size, 1, 1, seq_len] (广播后可用于注意力分数)
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def generate_subsequent_mask(sz, device='cpu'):
    """
    生成 Subsequent Mask (用于 Decoder 的自注意力，防止看到未来信息)
    Args:
        sz: 序列长度
    Returns:
        mask: [1, 1, sz, sz] 下三角矩阵
    """
    mask = torch.tril(torch.ones(sz, sz, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)

def create_masks(src, tgt, pad_idx=0, device='cpu'):
    """
    创建训练所需的全部 Mask
    """
    src_mask = generate_padding_mask(src, pad_idx).to(device)
    tgt_pad_mask = generate_padding_mask(tgt, pad_idx).to(device)
    tgt_len = tgt.size(1)
    tgt_sub_mask = generate_subsequent_mask(tgt_len, device)
    tgt_mask = tgt_pad_mask & tgt_sub_mask  # 结合两种 mask
    return src_mask, tgt_mask

# --------------------------
# 1. 位置编码
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --------------------------
# 2. 多头注意力
# --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q, k, v: [batch, n_heads, seq_len, d_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, v)
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性变换 + 分头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(attn_output)

# --------------------------
# 3. Feed Forward
# --------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# --------------------------
# 4. Encoder Layer
# --------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Self-Attention
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

# --------------------------
# 5. Decoder Layer
# --------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # Masked Self-Attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Cross-Attention
        cross_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(cross_out))

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x

# --------------------------
# 6. 完整 Encoder
# --------------------------
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        
        # 缩放 embedding
        self.scale = math.sqrt(d_model)

    def forward(self, x, src_mask):
        x = self.embedding(x) * self.scale
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

# --------------------------
# 7. 完整 Decoder
# --------------------------
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(d_model)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.embedding(x) * self.scale
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

# --------------------------
# 8. 最终 Transformer
# --------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx=0,
        tgt_pad_idx=0,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout, src_pad_idx)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout, tgt_pad_idx)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc(dec_out)

    def encode(self, src, src_mask=None):
        """单独调用 Encoder"""
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        """单独调用 Decoder"""
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc(dec_out)

# --------------------------
# 9. Noam 学习率调度器
# --------------------------
class NoamScheduler:
    """Transformer 论文中的学习率调度策略"""
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        arg1 = self.step_num ** (-0.5)
        arg2 = self.step_num * (self.warmup_steps ** (-1.5))
        return self.d_model ** (-0.5) * min(arg1, arg2)

# --------------------------
# 10. 数据集示例
# --------------------------
class TranslationDataset(Dataset):
    """简单的翻译数据集示例"""
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, max_len=128):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = self.src_data[idx][:self.max_len]
        tgt = self.tgt_data[idx][:self.max_len]
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0):
    """自定义 collate 函数，处理变长序列"""
    src_batch, tgt_batch = zip(*batch)
    
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
    
    return src_batch, tgt_batch

# --------------------------
# 11. 训练器
# --------------------------
class Trainer:
    def __init__(self, model, train_loader, val_loader=None, 
                 src_pad_idx=0, tgt_pad_idx=0, 
                 d_model=512, warmup_steps=4000,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
        self.optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = NoamScheduler(self.optimizer, d_model, warmup_steps)
        
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for src, tgt in pbar:
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # 准备输入输出
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 创建 masks
            src_mask, tgt_mask = create_masks(src, tgt_input, self.src_pad_idx, self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = self.criterion(output, tgt_output)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': self.scheduler.get_lr()})
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        
        for src, tgt in tqdm(self.val_loader, desc='Validating'):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask = create_masks(src, tgt_input, self.src_pad_idx, self.device)
            
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = self.criterion(output, tgt_output)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}', end='')
            if val_loss is not None:
                print(f', Val Loss = {val_loss:.4f}')
            else:
                print()

# --------------------------
# 12. 推理函数 (Greedy Decoding)
# --------------------------
@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device='cpu'):
    """
    贪心解码
    Args:
        model: Transformer 模型
        src: [1, src_len] 源语言序列
        src_mask: [1, 1, 1, src_len] 源语言 mask
        max_len: 最大生成长度
        start_symbol: 起始符号索引
        end_symbol: 结束符号索引
    """
    model.eval()
    
    # Encoder 前向传播
    enc_out = model.encode(src, src_mask)
    
    # 初始化 Decoder 输入
    ys = torch.ones(1, 1).fill_(start_symbol).long().to(device)
    
    for i in range(max_len - 1):
        # 创建 Decoder mask
        tgt_mask = generate_subsequent_mask(ys.size(1), device)
        
        # Decoder 前向传播
        out = model.decode(ys, enc_out, src_mask, tgt_mask)
        
        # 获取最后一个位置的预测
        prob = out[:, -1, :]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # 拼接
        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).long().to(device)], dim=1)
        
        if next_word == end_symbol:
            break
            
    return ys

# --------------------------
# 13. 完整测试示例
# --------------------------
def create_dummy_data(num_samples=1000, src_vocab_size=100, tgt_vocab_size=100, max_len=20):
    """创建虚拟数据用于测试"""
    src_data = []
    tgt_data = []
    
    for _ in range(num_samples):
        src_len = np.random.randint(5, max_len)
        tgt_len = np.random.randint(5, max_len)
        
        # 加入 BOS/EOS
        src = [np.random.randint(1, src_vocab_size) for _ in range(src_len)]
        tgt = [2] + [np.random.randint(3, tgt_vocab_size) for _ in range(tgt_len)] + [3]
        
        src_data.append(src)
        tgt_data.append(tgt)
    
    return src_data, tgt_data

if __name__ == "__main__":
    # 超参数
    SRC_VOCAB_SIZE = 100
    TGT_VOCAB_SIZE = 100
    D_MODEL = 256
    N_LAYERS = 3
    N_HEADS = 8
    D_FF = 512
    MAX_LEN = 128
    DROPOUT = 0.1
    BATCH_SIZE = 32
    EPOCHS = 10
    
    SRC_PAD_IDX = 0
    TGT_PAD_IDX = 0
    BOS_IDX = 2
    EOS_IDX = 3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建虚拟数据
    print("Creating dummy data...")
    src_data, tgt_data = create_dummy_data(1000, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    
    # 数据集和数据加载器
    train_dataset = TranslationDataset(src_data[:800], tgt_data[:800], None, None, MAX_LEN)
    val_dataset = TranslationDataset(src_data[800:], tgt_data[800:], None, None, MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, SRC_PAD_IDX, TGT_PAD_IDX)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, SRC_PAD_IDX, TGT_PAD_IDX)
    )
    
    # 初始化模型
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        src_pad_idx=SRC_PAD_IDX,
        tgt_pad_idx=TGT_PAD_IDX,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 训练
    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        src_pad_idx=SRC_PAD_IDX,
        tgt_pad_idx=TGT_PAD_IDX,
        d_model=D_MODEL,
        warmup_steps=400,
        device=device
    )
    
    trainer.train(EPOCHS)
    
    # 测试推理
    print("\nTesting inference...")
    model.eval()
    test_src = torch.randint(1, SRC_VOCAB_SIZE, (1, 10)).to(device)
    src_mask = generate_padding_mask(test_src, SRC_PAD_IDX).to(device)
    
    result = greedy_decode(
        model, test_src, src_mask, 
        max_len=MAX_LEN, 
        start_symbol=BOS_IDX, 
        end_symbol=EOS_IDX,
        device=device
    )
    
    print(f"Input shape: {test_src.shape}")
    print(f"Input sequence: {test_src[0].tolist()}")
    print(f"Output shape: {result.shape}")
    print(f"Output sequence: {result[0].tolist()}")
    print("\n✅ Transformer 完整实现测试成功！")
