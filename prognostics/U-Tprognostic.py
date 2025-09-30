import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
torch.backends.cudnn.benchmark = True

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=5, feat_dim=256):
        super().__init__()
        self.feat_dim = feat_dim
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return e3

class CNNDecoder(nn.Module):
    def __init__(self, feat_dim=256, n_classes=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.up2 = nn.ConvTranspose2d(feat_dim, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d2 = self.up2(x)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = self.dec1(d1)
        out = self.out_conv(d1)
        return out

class SeqTransformer(nn.Module):
    def __init__(self, feat_dim=256, max_seq_len=5, n_layers=2, n_heads=8, dim_ff=512):
        super().__init__()
        self.feat_dim = feat_dim
        self.max_seq_len = max_seq_len
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, feat_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, feats):
        Bp, L, D = feats.shape
        pe = self.pos_embedding[:L, :].unsqueeze(0).expand(Bp, -1, -1)
        x = feats + pe
        out = self.transformer(x)
        return out[:, -1, :]

class SeqViTModel(nn.Module):
    def __init__(self,
                 in_channels=5,
                 feat_dim=256,
                 max_seq_len=5,
                 n_layers=2,
                 n_heads=8,
                 dim_ff=512,
                 n_classes=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.max_seq_len = max_seq_len
        self.encoder = CNNEncoder(in_channels=in_channels, feat_dim=feat_dim)
        self.transformer = SeqTransformer(
            feat_dim=feat_dim,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_ff=dim_ff
        )
        self.decoder = CNNDecoder(feat_dim=feat_dim, n_classes=n_classes)

    def forward(self, seq_inp):
        B, L, C, H, W = seq_inp.shape
        device = seq_inp.device
        flat_inp = seq_inp.view(B * L, C, H, W)
        feats = self.encoder(flat_inp)
        H4, W4 = H // 4, W // 4
        feats = feats.view(B, L, self.feat_dim, H4, W4)
        feats = feats.permute(0, 3, 4, 1, 2).contiguous()
        feats = feats.view(B * H4 * W4, L, self.feat_dim)
        pred_feats = self.transformer(feats)
        pred_feats = pred_feats.view(B, H4, W4, self.feat_dim).permute(0, 3, 1, 2).contiguous()
        out_logits = self.decoder(pred_feats)
        out_logits = F.interpolate(out_logits, size=(H, W), mode='bilinear', align_corners=False)
        return out_logits

class SeqTrainDataset(Dataset):
    def __init__(self, volumes, cycles, max_seq_len):
        self.vols = volumes
        self.cycles = cycles
        self.T, self.H, self.W = volumes.shape
        self.max_diff = float(cycles[-1] - cycles[0])
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.T - 1 - (self.max_seq_len - 1)

    def __getitem__(self, idx):
        t = idx + (self.max_seq_len - 1)
        frames = []
        for i in range(t - self.max_seq_len + 1, t + 1):
            img = self.vols[i]
            one_hot = np.eye(4)[img].transpose(2, 0, 1).astype(np.float32)
            if i > 0:
                delta_val = float(self.cycles[i] - self.cycles[i - 1]) / self.max_diff
            else:
                delta_val = 0.0
            delta = np.full((1, self.H, self.W), delta_val, dtype=np.float32)
            frames.append(np.concatenate([one_hot, delta], axis=0))
        seq_np = np.stack(frames, axis=0)
        tgt = self.vols[t + 1].astype(np.int64)
        inp_depth = self.vols[t].astype(np.int64)
        return torch.from_numpy(seq_np), torch.from_numpy(tgt), torch.from_numpy(inp_depth)

class IterTestDataset(Dataset):
    def __init__(self, volumes, cycles, max_seq_len):
        self.vols = volumes
        self.cycles = cycles
        self.T, self.H, self.W = volumes.shape
        self.max_diff = float(cycles[-1] - cycles[0])
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.T - 1

    def __getitem__(self, t_idx):
        frames = []
        for t in range(0, t_idx + 1):
            img_t = self.vols[t]
            one_hot = np.eye(4)[img_t].transpose(2, 0, 1).astype(np.float32)
            delta_val = float(self.cycles[t] - self.cycles[0]) / self.max_diff
            h, w = self.vols.shape[1], self.vols.shape[2]
            delta = np.full((1, h, w), delta_val, dtype=np.float32)
            frames.append(np.concatenate([one_hot, delta], axis=0))
        L_real = t_idx + 1
        if L_real < self.max_seq_len:
            pad_count = self.max_seq_len - L_real
            h, w = self.vols.shape[1], self.vols.shape[2]
            zero_frame = np.zeros((5, h, w), dtype=np.float32)
            frames = [zero_frame] * pad_count + frames
            seq_np = np.stack(frames, axis=0)
            seq_len = L_real
        else:
            seq_np = np.stack(frames[-self.max_seq_len:], axis=0)
            seq_len = self.max_seq_len
        tgt = self.vols[-1].astype(np.int64)
        return torch.from_numpy(seq_np), torch.from_numpy(tgt), t_idx, seq_len

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

torch.backends.cudnn.benchmark = True

def labels_to_bgr(labels):
    h, w = labels.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[labels == 0] = [193,182,255]
    bgr[labels == 1] = [0,255,255]
    bgr[labels == 2] = [255,255,0]
    bgr[labels == 3] = [128,128,128]
    return bgr

def cross_validate(data_dir, epochs, batch_size, lr,
                   max_seq_len, hidden_dim,
                   shape_w_dice, shape_w_jaccard,
                   results_dir, model_out_dir, log_out):
    files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
    assert len(files) >= 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(log_out), exist_ok=True)
    f_log = open(log_out, 'w')
    def tprint(*vals, **kwargs):
        print(*vals, **kwargs)
        print(*vals, **kwargs, file=f_log)
    tprint(f"Device: {device}, Params: max_seq={max_seq_len}, hid={hidden_dim}, "
           f"sd={shape_w_dice}, sj={shape_w_jaccard}, lr={lr}, bs={batch_size}\n")
    params_name = f"seq{max_seq_len}_hd{hidden_dim}_sd{shape_w_dice}_sj{shape_w_jaccard}_" \
                  f"lr{lr}_bs{batch_size}"
    base_out = os.path.join(results_dir, params_name)
    os.makedirs(base_out, exist_ok=True)
    tprint(f"Results under: {base_out}\n")
    all_fold_accs = []
    for fold_idx, test_path in enumerate(files, start=1):
        tprint(f"--- Fold {fold_idx}/{len(files)} ---")
        train_paths = [p for p in files if p != test_path]
        train_ds = []
        for p in train_paths:
            raw = np.load(p, allow_pickle=True).item()
            cycles = sorted(int(k) for k in raw.keys())
            vols = np.stack([raw[str(c)] for c in cycles], axis=0)
            train_ds.append(SeqTrainDataset(vols, cycles, max_seq_len))
        train_loader = DataLoader(
            ConcatDataset(train_ds),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        raw = np.load(test_path, allow_pickle=True).item()
        cycles = sorted(int(k) for k in raw.keys())
        vols = np.stack([raw[str(c)] for c in cycles], axis=0)
        test_ds = IterTestDataset(vols, cycles, max_seq_len)
        test_loader = DataLoader(
            test_ds,
            batch_size=1, shuffle=False,
            num_workers=2, pin_memory=True
        )
        model = SeqViTModel(
            in_channels=5,
            feat_dim=hidden_dim,
            max_seq_len=max_seq_len,
            n_layers=2,
            n_heads=8,
            dim_ff=512,
            n_classes=4
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        ce_criterion = nn.CrossEntropyLoss(reduction='none')
        scaler = GradScaler()
        fold_dir = os.path.join(base_out, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        tprint(f"Fold {fold_idx} outputs to: {fold_dir}\n")
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for seq_inp, tgt, inp_depth in train_loader:
                seq_inp, tgt, inp_depth = seq_inp.to(device), tgt.to(device), inp_depth.to(device)
                optimizer.zero_grad()
                with autocast():
                    out_logits = model(seq_inp)
                    ce_map = ce_criterion(out_logits, tgt)
                    last_one_hot = seq_inp[:, -1, :4].sum(dim=1)
                    inp_mask = (last_one_hot > 0).float()
                    weight_map = 1.0 + 0.5 * inp_mask
                    ce_loss = (ce_map * weight_map).mean()
                    probs = torch.softmax(out_logits, dim=1)
                    pred_binary_prob = probs[:, 1:].sum(dim=1)
                    gt_binary = (tgt > 0).float()
                    Bn = pred_binary_prob.size(0)
                    pf = pred_binary_prob.view(Bn, -1)
                    tf = gt_binary.view(Bn, -1)
                    inter = (pf * tf).sum(dim=1)
                    sum_p = pf.sum(dim=1)
                    sum_t = tf.sum(dim=1)
                    dice = (2 * inter + 1e-6) / (sum_p + sum_t + 1e-6)
                    d_loss = 1 - dice.mean()
                    union = sum_p + sum_t - inter
                    jaccard = (inter + 1e-6) / (union + 1e-6)
                    j_loss = 1 - jaccard.mean()
                    shape_loss = shape_w_dice * d_loss + shape_w_jaccard * j_loss
                    pred_mask = (out_logits.argmax(dim=1) > 0).float()
                    shape_mono = torch.relu(inp_mask - pred_mask).mean()
                    pred_depth = out_logits.argmax(dim=1).float()
                    inp_depth_f = inp_depth.float()
                    depth_mono = torch.relu(inp_depth_f - pred_depth).mean()
                    loss = ce_loss + shape_loss + 0.5 * shape_mono + 0.5 * depth_mono
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            tprint(f"Fold{fold_idx} Epoch{epoch:2d} ▶︎ Train Loss: {avg_train_loss:.4f}")
        fold_model_path = os.path.join(model_out_dir, f"{params_name}_fold{fold_idx}.pth")
        os.makedirs(os.path.dirname(fold_model_path), exist_ok=True)
        torch.save(model.state_dict(), fold_model_path)
        tprint(f"Saved model to {fold_model_path}\n")
        model.eval()
        T = vols.shape[0]
        H, W = vols.shape[1], vols.shape[2]
        all_preds_acc = []
        with torch.no_grad():
            for t_idx in range(0, T - 1):
                frames = []
                for t in range(0, t_idx + 1):
                    img_t = vols[t]
                    one_hot = np.eye(4)[img_t].transpose(2, 0, 1).astype(np.float32)
                    delta_val = float(cycles[t] - cycles[0]) / float(cycles[-1] - cycles[0])
                    delta = np.full((1, H, W), delta_val, dtype=np.float32)
                    frames.append(np.concatenate([one_hot, delta], axis=0))
                L_real = t_idx + 1
                if L_real < max_seq_len:
                    pad_count = max_seq_len - L_real
                    zero_frame = np.zeros((5, H, W), dtype=np.float32)
                    frames = [zero_frame] * pad_count + frames
                else:
                    frames = frames[-max_seq_len:]
                last_one_hot = frames[-1][:4]
                inp_labels = last_one_hot.argmax(axis=0)
                current_mask = inp_labels.copy()
                preds_sequence = []
                base_frames = [f.copy() for f in frames]
                for step in range((T - 1) - t_idx):
                    stacked = np.stack(base_frames, axis=0)
                    inp_seq = torch.from_numpy(stacked).unsqueeze(0).to(device)
                    out_logits = model(inp_seq)
                    out_pred = out_logits[0].argmax(dim=0).cpu().numpy().astype(np.int64)
                    current_mask = np.maximum(current_mask, out_pred)
                    preds_sequence.append(current_mask.copy())
                    one_hot_next = np.eye(4)[current_mask].transpose(2, 0, 1).astype(np.float32)
                    idx_now = t_idx + step + 1
                    delta_val = float(cycles[idx_now] - cycles[0]) / float(cycles[-1] - cycles[0])
                    delta = np.full((1, H, W), delta_val, dtype=np.float32)
                    frame_np = np.concatenate([one_hot_next, delta], axis=0)
                    if len(base_frames) == max_seq_len:
                        base_frames.pop(0)
                    base_frames.append(frame_np)
                accs = []
                for step_idx, pred_mask in enumerate(preds_sequence):
                    true_mask = vols[t_idx + 1 + step_idx]
                    correct = (pred_mask == true_mask).sum()
                    total = true_mask.size
                    accs.append(correct / total)
                all_preds_acc.append(accs)
                num_future = len(preds_sequence)
                fig, axes = plt.subplots(1, num_future + 1, figsize=(3 * (num_future + 1), 3))
                inp_bgr = labels_to_bgr(inp_labels)
                inp_rgb = cv2.cvtColor(inp_bgr, cv2.COLOR_BGR2RGB)
                axes[0].imshow(inp_rgb)
                axes[0].set_title(f"Input t={t_idx}")
                axes[0].axis('off')
                for k in range(num_future):
                    pred_mask = preds_sequence[k]
                    pred_bgr = labels_to_bgr(pred_mask)
                    pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
                    ax = axes[k + 1]
                    ax.imshow(pred_rgb)
                    ax.set_title(f"Pred t={t_idx + 1 + k}")
                    ax.axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(fold_dir, f"fold{fold_idx}_t{t_idx}_color_all.png"))
                plt.close(fig)
        max_future = max(len(a) for a in all_preds_acc)
        acc_matrix = np.zeros((len(all_preds_acc), max_future), dtype=np.float32)
        for i, accs in enumerate(all_preds_acc):
            acc_matrix[i, :len(accs)] = accs
        np.savetxt(os.path.join(fold_dir, "acc_matrix.csv"), acc_matrix, delimiter=",")
        plt.figure(figsize=(8, 6))
        plt.imshow(acc_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Accuracy')
        plt.xlabel('Prediction Step (relative)')
        plt.ylabel('Starting Frame t_idx')
        plt.title(f'Fold{fold_idx} Accuracy Heatmap')
        plt.xticks(ticks=np.arange(max_future), labels=[f"{i+1}" for i in range(max_future)])
        plt.yticks(ticks=np.arange(len(all_preds_acc)), labels=[f"{i}" for i in range(len(all_preds_acc))])
        plt.savefig(os.path.join(fold_dir, f"fold{fold_idx}_acc_heatmap.png"))
        plt.close()
        mean_acc = acc_matrix.mean()
        tprint(f"Fold{fold_idx} mean accuracy over all steps: {mean_acc:.4f}\n")
        all_fold_accs.append(acc_matrix)
    mean_accs = [mat.mean() for mat in all_fold_accs]
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(mean_accs) + 1), mean_accs)
    plt.xlabel('Fold')
    plt.ylabel('Mean Accuracy (all steps)')
    plt.title('Cross-Validation Mean Accuracy per Fold')
    plt.grid(axis='y')
    plt.savefig(os.path.join(base_out, 'mean_accuracy_per_fold.png'))
    plt.close()
    tprint("\n=== All folds complete ===")
    for i, m in enumerate(mean_accs, start=1):
        tprint(f" Fold {i} Mean Acc: {m:.4f}")
    f_log.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       type=str, required=True)
    parser.add_argument('--epochs',         type=int, required=True)
    parser.add_argument('--batch',          type=int, required=True)
    parser.add_argument('--lr',             type=float, required=True)
    parser.add_argument('--max_seq',        type=int, required=True)
    parser.add_argument('--hidden_dim',     type=int, required=True)
    parser.add_argument('--shape_w_dice',   type=float, required=True)
    parser.add_argument('--shape_w_jaccard',type=float, required=True)
    parser.add_argument('--results_dir',    type=str, required=True)
    parser.add_argument('--model_out_dir',  type=str, required=True)
    parser.add_argument('--log_out',        type=str, required=True)
    args = parser.parse_args()
    cross_validate(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_seq_len=args.max_seq,
        hidden_dim=args.hidden_dim,
        shape_w_dice=args.shape_w_dice,
        shape_w_jaccard=args.shape_w_jaccard,
        results_dir=args.results_dir,
        model_out_dir=args.model_out_dir,
        log_out=args.log_out
    )
