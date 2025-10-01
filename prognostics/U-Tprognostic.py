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
    bgr[labels == 0] = [255,255,255]
    bgr[labels == 1] = [0,255,255]
    bgr[labels == 2] = [255,255,0]
    bgr[labels == 3] = [128,128,128]
    return bgr

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
        pe = self.pos_embedding[:L].unsqueeze(0).expand(Bp, -1, -1)
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
        flat = seq_inp.view(B*L, C, H, W)
        feats = self.encoder(flat)
        H4, W4 = H//4, W//4
        feats = feats.view(B, L, self.feat_dim, H4, W4)
        feats = feats.permute(0,3,4,1,2).contiguous().view(B*H4*W4, L, self.feat_dim)
        pred_feats = self.transformer(feats)
        pred_feats = pred_feats.view(B, H4, W4, self.feat_dim).permute(0,3,1,2).contiguous()
        out_logits = self.decoder(pred_feats)
        out_logits = F.interpolate(out_logits, size=(H,W), mode='bilinear', align_corners=False)
        return out_logits

class SeqTrainDataset(Dataset):
    def __init__(self, volumes, cycles, max_seq_len):
        self.vols = volumes
        self.cycles = cycles
        self.T, self.H, self.W = volumes.shape
        self.max_diff = float(cycles[-1] - cycles[0])
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.T - self.max_seq_len

    def __getitem__(self, idx):
        t = idx + self.max_seq_len - 1
        frames = []
        for i in range(t - self.max_seq_len + 1, t + 1):
            img = self.vols[i]
            one_hot = np.eye(4)[img].transpose(2,0,1).astype(np.float32)
            delta = float(self.cycles[i] - self.cycles[0]) / self.max_diff
            delta_map = np.full((1,self.H,self.W), delta, np.float32)
            frames.append(np.concatenate([one_hot, delta_map], axis=0))

        future_delta = float(self.cycles[t + 1] - self.cycles[0]) / self.max_diff
        frames[-1][4, :, :] = future_delta
        seq_np = np.stack(frames, axis=0)
        tgt = self.vols[t+1].astype(np.int64)
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
        for t in range(t_idx+1):
            img = self.vols[t]
            one_hot = np.eye(4)[img].transpose(2,0,1).astype(np.float32)
            delta = float(self.cycles[t] - self.cycles[0]) / self.max_diff
            delta_map = np.full((1,self.H,self.W), delta, np.float32)
            frames.append(np.concatenate([one_hot, delta_map], axis=0))
        L_real = t_idx + 1
        if L_real < self.max_seq_len:
            pad = [np.zeros((5,self.H,self.W),np.float32)]*(self.max_seq_len - L_real)
            frames = pad + frames
        else:
            frames = frames[-self.max_seq_len:]
        tgt = self.vols[-1].astype(np.int64)
        return torch.from_numpy(np.stack(frames,axis=0)), torch.from_numpy(tgt), t_idx, min(L_real, self.max_seq_len)

def cross_validate(data_dir, epochs, batch_size, lr,
                   max_seq_len, hidden_dim,
                   w_ce, w_dice, w_jaccard,
                   w_shape_mono, w_depth_mono, w_expand,w_focus,
                   results_dir, model_out_dir, log_out):
    files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
    assert len(files) >= 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(os.path.dirname(log_out), exist_ok=True)
    f_log = open(log_out, 'w')
    def tprint(*vals, **kwargs):
        print(*vals, **kwargs)
        print(*vals, **kwargs, file=f_log)

    tprint(f"Device: {device}, Params: seq={max_seq_len}, hid={hidden_dim}, lr={lr}, bs={batch_size}\n")

    params_name = f"seq{max_seq_len}_hd{hidden_dim}_lr{lr}_bs{batch_size}"
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
        train_loader = DataLoader(ConcatDataset(train_ds), batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)

        raw = np.load(test_path, allow_pickle=True).item()
        cycles = sorted(int(k) for k in raw.keys())
        vols = np.stack([raw[str(c)] for c in cycles], axis=0)
        test_ds = IterTestDataset(vols, cycles, max_seq_len)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                 num_workers=2, pin_memory=True)

        model = SeqViTModel(in_channels=5, feat_dim=hidden_dim,
                            max_seq_len=max_seq_len, n_layers=2,
                            n_heads=8, dim_ff=512, n_classes=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        ce_criterion = nn.CrossEntropyLoss(reduction='none')
        scaler = GradScaler()

        fold_dir = os.path.join(base_out, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        tprint(f"Fold {fold_idx} outputs to: {fold_dir}\n")

        for epoch in range(1, epochs + 1):
            model.train()
            sum_ce = sum_dice = sum_jacc = sum_shape = sum_depth = sum_expand = 0.0
            total_loss = 0.0
            for seq_inp, tgt, inp_depth in train_loader:
                seq_inp, tgt, inp_depth = seq_inp.to(device), tgt.to(device), inp_depth.to(device)
                optimizer.zero_grad()
                with autocast():
                    out_logits = model(seq_inp)
                    H_in, W_in = seq_inp.shape[3], seq_inp.shape[4]
                    if out_logits.shape[2:] != (H_in, W_in):
                        out_logits = F.interpolate(out_logits, size=(H_in, W_in),
                                                   mode='bilinear', align_corners=False)

                    ce_map = ce_criterion(out_logits, tgt)
                    last_one_hot = seq_inp[:, -1, :4].sum(dim=1)
                    inp_mask = (last_one_hot > 0).float()
                    ce_loss = (ce_map * (1.0 + w_focus * inp_mask)).mean()
                    d_logits = torch.softmax(out_logits, dim=1)[:,1:].sum(dim=1)
                    gt_bin = (tgt>0).float()
                    pf, tf = d_logits.view(d_logits.size(0),-1), gt_bin.view(gt_bin.size(0),-1)
                    inter = (pf * tf).sum(1); sum_p, sum_t = pf.sum(1), tf.sum(1)
                    dice = (2*inter+1e-6)/(sum_p+sum_t+1e-6); d_loss = 1-dice.mean()
                    union = sum_p+sum_t-inter; jacc = (inter+1e-6)/(union+1e-6); j_loss = 1-jacc.mean()
                    pred_bin = (out_logits.argmax(1)>0).float()
                    shape_mono = torch.relu(inp_mask - pred_bin).mean()
                    depth_mono = torch.relu(inp_depth.float() - out_logits.argmax(1).float()).mean()
                    gt_dil = F.max_pool2d(gt_bin.unsqueeze(1), kernel_size=(1,3), padding=(0,1)).squeeze(1)
                    if pred_bin.shape[1:] != gt_dil.shape[1:]:
                        pred_bin = F.interpolate(pred_bin.unsqueeze(1),
                                                 size=gt_dil.shape[-2:],
                                                 mode='nearest').squeeze(1)
                    horiz_expand = torch.relu(gt_dil - pred_bin).mean()

                    loss = (w_ce*ce_loss + w_dice*d_loss + w_jaccard*j_loss
                            + w_shape_mono*shape_mono + w_depth_mono*depth_mono
                            + w_expand*horiz_expand)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                sum_ce    += ce_loss.item()
                sum_dice  += d_loss.item()
                sum_jacc  += j_loss.item()
                sum_shape += shape_mono.item()
                sum_depth += depth_mono.item()
                sum_expand+= horiz_expand.item()
                total_loss+= loss.item()
                
            n_batches = len(train_loader)
            avg_ce    = sum_ce    / n_batches
            avg_dice  = sum_dice  / n_batches
            avg_jacc  = sum_jacc  / n_batches
            avg_shape = sum_shape / n_batches
            avg_depth = sum_depth / n_batches
            avg_expand= sum_expand/ n_batches
            avg_total = total_loss/ n_batches
                
            tprint(f"Fold{fold_idx} Epoch{epoch:2d} CE: {avg_ce:.4f}")
        model_path = os.path.join(model_out_dir, f"{params_name}_fold{fold_idx}.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        tprint(f"Saved model to {model_path}\n")

        model.eval()
        T, H, W = vols.shape
        all_preds_acc=[]
        with torch.no_grad():
            for t_idx in range(T - 1):
                t_dir = os.path.join(fold_dir, f"t{t_idx}")
                os.makedirs(t_dir, exist_ok=True)

                frames = []
                for t in range(t_idx + 1):
                    one_hot = np.eye(4)[vols[t]].transpose(2,0,1).astype(np.float32)
                    delta = float(cycles[t] - cycles[0]) / float(cycles[-1] - cycles[0])
                    delta_map = np.full((1, H, W), delta, np.float32)
                    frames.append(np.concatenate([one_hot, delta_map], axis=0))

                if len(frames) < max_seq_len:
                    pad = [np.zeros((5,H,W),np.float32)] * (max_seq_len - len(frames))
                    frames = pad + frames
                else:
                    frames = frames[-max_seq_len:]

                inp_labels = frames[-1][:4].argmax(axis=0)
                np.savetxt(os.path.join(t_dir, "input_labels.csv"),
                           inp_labels, delimiter=",", fmt="%d")

                base = [f.copy() for f in frames]
                preds_seq = []
                for step in range((T - 1) - t_idx):
                    
                    

                    target_idx = t_idx + step + 1
                    phi_target = float(cycles[target_idx] - cycles[0]) / float(cycles[-1] - cycles[0])

                    temp_base = [f.copy() for f in base]
                    temp_base[-1][4, :, :] = phi_target
                    
                    
                    
                    
                    
                    
                    seq_np = np.stack(base, axis=0)
                    inp_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device)
                    out = model(inp_tensor)
                    out = F.interpolate(out, size=(H,W), mode='bilinear', align_corners=False)
                    pred = out[0].argmax(dim=0).cpu().numpy().astype(np.int64)
                    if step == 0:
                        cur_mask = inp_labels.copy()
                    cur_mask = np.maximum(cur_mask, pred)
                    preds_seq.append(cur_mask.copy())

                    onehot = np.eye(4)[cur_mask].transpose(2,0,1).astype(np.float32)
                    delta = float(cycles[target_idx] - cycles[0]) / float(cycles[-1] - cycles[0])
                    delta_map = np.full((1,H,W), delta, np.float32)
                    next_frame = np.concatenate([onehot, delta_map], axis=0)
                    base = (base + [next_frame])[-max_seq_len:]

                for i, mask in enumerate(preds_seq, start=1):
                    np.savetxt(
                        os.path.join(t_dir, f"pred_labels_step{i}.csv"),
                        mask, delimiter=",", fmt="%d"
                    )

                num = len(preds_seq)
                fig, axes = plt.subplots(1, num+1, figsize=(3*(num+1),3))
                inp_bgr = labels_to_bgr(inp_labels)
                axes[0].imshow(cv2.cvtColor(inp_bgr, cv2.COLOR_BGR2RGB))
                axes[0].set_title(f"Input t={t_idx}"); axes[0].axis('off')
                for k, mask in enumerate(preds_seq):
                    bgr = labels_to_bgr(mask)
                    ax = axes[k+1]
                    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"Pred t={t_idx + k +1}"); ax.axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(t_dir, "color_all.png"))
                plt.close(fig)
                accs = []
                for step_idx, pred_mask in enumerate(preds_seq):
                    true_mask = vols[t_idx + 1 + step_idx]
                    correct = (pred_mask == true_mask).sum()
                    total = true_mask.size
                    accs.append(correct / total)
                all_preds_acc.append(accs)

        max_future = max(len(a) for a in all_preds_acc)
        acc_matrix = np.zeros((len(all_preds_acc), max_future), dtype=np.float32)
        for i, accs in enumerate(all_preds_acc):
            acc_matrix[i, :len(accs)] = accs

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
        all_fold_accs.append(acc_matrix)
        

        csv_path = os.path.join(fold_dir, "acc_matrix.csv")
        np.savetxt(csv_path, acc_matrix, delimiter=",")

    mean_accs = [mat.mean() for mat in all_fold_accs]
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(mean_accs) + 1), mean_accs)
    plt.xlabel('Fold')
    plt.ylabel('Mean Accuracy (all steps)')
    plt.title('Cross-Validation Mean Accuracy per Fold')
    plt.grid(axis='y')
    plt.savefig(os.path.join(fold_dir, 'mean_accuracy_per_fold.png'))
    plt.close()

    tprint("\n=== All folds complete ===")
    for i, m in enumerate(mean_accs, start=1):
        tprint(f" Fold {i} Mean Acc: {m:.4f}")
    f_log.close()

def run_inference(npy_file, model_file, max_seq_len, hidden_dim, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = np.load(npy_file, allow_pickle=True).item()
    cycles = sorted(int(k) for k in data.keys())
    vols = np.stack([data[str(c)] for c in cycles], axis=0)
    T, H, W = vols.shape

    model = SeqViTModel(in_channels=5, feat_dim=hidden_dim,
                        max_seq_len=max_seq_len, n_layers=2,
                        n_heads=8, dim_ff=512, n_classes=4).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    with torch.no_grad():
        for t_idx in range(T - 1):
            t_dir = os.path.join(output_dir, f"t{t_idx}")
            os.makedirs(t_dir, exist_ok=True)

            frames = []
            for t in range(t_idx + 1):
                one_hot = np.eye(4)[vols[t]].transpose(2,0,1).astype(np.float32)
                delta = float(cycles[t] - cycles[0]) / float(cycles[-1] - cycles[0])
                delta_map = np.full((1, H, W), delta, np.float32)
                frames.append(np.concatenate([one_hot, delta_map], axis=0))

            if len(frames) < max_seq_len:
                pad = [np.zeros((5,H,W),np.float32)] * (max_seq_len - len(frames))
                frames = pad + frames
            else:
                frames = frames[-max_seq_len:]

            inp_labels = frames[-1][:4].argmax(axis=0)
            np.savetxt(os.path.join(t_dir, "input_labels.csv"),
                       inp_labels, delimiter=",", fmt="%d")

            base = [f.copy() for f in frames]
            preds_seq = []
            for step in range((T - 1) - t_idx):
                target_idx = t_idx + step + 1
                phi_target = float(cycles[target_idx] - cycles[0]) / float(cycles[-1] - cycles[0])

                temp_base = [f.copy() for f in base]
                temp_base[-1][4, :, :] = phi_target
                seq_np = np.stack(base, axis=0)
                inp_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device)
                out = model(inp_tensor)
                out = F.interpolate(out, size=(H,W), mode='bilinear', align_corners=False)
                pred = out[0].argmax(dim=0).cpu().numpy().astype(np.int64)
                if step == 0:
                    cur_mask = inp_labels.copy()
                cur_mask = np.maximum(cur_mask, pred)
                preds_seq.append(cur_mask.copy())

                onehot = np.eye(4)[cur_mask].transpose(2,0,1).astype(np.float32)
                delta = float(cycles[target_idx] - cycles[0]) / float(cycles[-1] - cycles[0])
                delta_map = np.full((1,H,W), delta, np.float32)
                next_frame = np.concatenate([onehot, delta_map], axis=0)
                base = (base + [next_frame])[-max_seq_len:]

            for i, mask in enumerate(preds_seq, start=1):
                np.savetxt(
                    os.path.join(t_dir, f"pred_labels_step{i}.csv"),
                    mask, delimiter=",", fmt="%d"
                )

            num = len(preds_seq)
            fig, axes = plt.subplots(1, num+1, figsize=(3*(num+1),3))
            inp_bgr = labels_to_bgr(inp_labels)
            axes[0].imshow(cv2.cvtColor(inp_bgr, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"Input t={t_idx}"); axes[0].axis('off')
            for k, mask in enumerate(preds_seq):
                bgr = labels_to_bgr(mask)
                ax = axes[k+1]
                ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Pred t={t_idx + k +1}"); ax.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(t_dir, "color_all.png"))
            plt.close(fig)

            inp_path = os.path.join(t_dir, f"input_t{t_idx:02d}.png")
            cv2.imwrite(inp_path, inp_bgr)

            for i, mask in enumerate(preds_seq, start=1):
                bgr = labels_to_bgr(mask)
                pred_path = os.path.join(t_dir, f"pred_t{t_idx:02d}_step{i:02d}.png")
                cv2.imwrite(pred_path, bgr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_train = subparsers.add_parser("train")
    p_train.add_argument('--data_dir', type=str, required=True)
    p_train.add_argument('--epochs', type=int, required=True)
    p_train.add_argument('--batch', type=int, required=True)
    p_train.add_argument('--lr', type=float, required=True)
    p_train.add_argument('--max_seq', type=int, required=True)
    p_train.add_argument('--hidden_dim', type=int, required=True)
    p_train.add_argument('--w_ce', type=float, required=True)
    p_train.add_argument('--w_dice', type=float, required=True)
    p_train.add_argument('--w_jaccard', type=float, required=True)
    p_train.add_argument('--w_shape_mono', type=float, required=True)
    p_train.add_argument('--w_depth_mono', type=float, required=True)
    p_train.add_argument('--w_expand', type=float, required=True)
    p_train.add_argument('--results_dir', type=str, required=True)
    p_train.add_argument('--model_out_dir', type=str, required=True)
    p_train.add_argument('--log_out', type=str, required=True)
    p_train.add_argument('--w_focus', type=float, default=0.5)
    p_pred = subparsers.add_parser("predict")
    p_pred.add_argument('--npy_file', type=str, required=True)
    p_pred.add_argument('--model_file', type=str, required=True)
    p_pred.add_argument('--max_seq', type=int, required=True)
    p_pred.add_argument('--hidden_dim', type=int, required=True)
    p_pred.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    if args.mode == "train":
        cross_validate(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            max_seq_len=args.max_seq,
            hidden_dim=args.hidden_dim,
            w_ce=args.w_ce,
            w_dice=args.w_dice,
            w_jaccard=args.w_jaccard,
            w_shape_mono=args.w_shape_mono,
            w_depth_mono=args.w_depth_mono,
            w_expand=args.w_expand,
            w_focus=args.w_focus,  
            results_dir=args.results_dir,
            model_out_dir=args.model_out_dir,
            log_out=args.log_out
        )
    else:
        run_inference(
            npy_file=args.npy_file,
            model_file=args.model_file,
            max_seq_len=args.max_seq,
            hidden_dim=args.hidden_dim,
            output_dir=args.output_dir
        )
