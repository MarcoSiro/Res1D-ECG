import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import wfdb
from torch.utils.data import Dataset, DataLoader

class ResidualBlock1D(nn.Module):   #ResBlock for ResNet1D
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
    
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:   
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
            
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv_path(x)   #Residual path   
        out += self.shortcut(x)      
        return self.relu(out)

class ResNet1D(nn.Module):       #ResNet1D with ResBlock
    def __init__(self, input_channels, num_classes):
        super(ResNet1D, self).__init__()
        
        self.prep = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = ResidualBlock1D(in_channels=64, out_channels=128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [batch, 12, 1000]
        x = self.prep(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x



class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate, num_classes=5):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.num_classes = num_classes

        self.save_hyperparameters(ignore=["model"])

        # Valutation metric: AUROC Multi-label
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_classes)
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_classes)
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_classes)

        # Clinical validation: F1 score 
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='macro')

        # ROC curve for clinical validation for test set
        self.test_roc = torchmetrics.ROC(task="multilabel", num_labels=num_classes)

        self.roc_data_saved = None
        self.final_auroc_saved = None
        self.final_f1_saved = None

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features) #logits output

        loss = F.binary_cross_entropy_with_logits(logits, true_labels) #loss calculation
        
        return loss, true_labels, logits

    def training_step(self, batch, batch_idx):
        loss, true_labels, logits = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_auroc(logits, true_labels.long())
        self.log("train_auroc", self.train_auroc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, logits = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        
        self.val_auroc(logits, true_labels.long())
        self.val_f1(logits, true_labels.long())
        
        self.log("val_auroc", self.val_auroc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, logits = self._shared_step(batch)
        
        self.test_auroc(logits, true_labels.long())
        self.test_f1(logits, true_labels.long())
        
        # Passiamo i dati alla funzione ROC per accumularli
        self.test_roc(logits, true_labels.long())
        
        self.log("test_auroc", self.test_auroc)
        self.log("test_f1", self.test_f1)

    def configure_optimizers(self):
        # Using AdamW optimazer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def on_test_epoch_end(self): #save the metrics
        self.roc_data_saved = self.test_roc.compute()
        self.final_auroc_saved = self.test_auroc.compute().item()
        self.final_f1_saved = self.test_f1.compute().item()
    



class PTBXLDataset(Dataset):  #Personalized Dataset for PTB-XL
    def __init__(self, df, data_path, all_classes): #df is the dataframe with the metadata, data_path is the path to the ECG files, all_classes is the list of all possible classes
        self.df = df
        self.data_path = data_path
        self.all_classes = all_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx] #Get the row of the dataframe corresponding to the index
        
        record = wfdb.rdrecord(f"{self.data_path}/{row.filename_lr}") #Read the ECG signal using wfdb, filename_lr is the column in the dataframe that contains the filename of the ECG signal
        signal = record.p_signal  #Get the signal from the record, p_signal is the attribute of the record that contains the signal data, shape is (1000, 12) because we have 1000 samples and 12 leads
        
        signal = signal.transpose() #Transpose the signal to have shape (12, 1000) for the Conv1D input
        
        mean = np.mean(signal, axis=1, keepdims=True) #Normalizing for stable training, mean and std are calculated for each lead (axis=1) and keepdims=True to keep the dimensions for broadcasting
        std = np.std(signal, axis=1, keepdims=True) + 1e-8
        signal = (signal - mean) / std
        
        signal_tensor = torch.tensor(signal, dtype=torch.float32) 
        
        labels = row['diagnostic_labels']     #Extracting labels
        label_tensor = torch.zeros(len(self.all_classes), dtype=torch.float32)   #Creating a multi-hot tensor for the labels
        for i, cls in enumerate(self.all_classes):
            if cls in labels:
                label_tensor[i] = 1.0
                
        return signal_tensor, label_tensor
    

    

class PTBXLDataModule(L.LightningDataModule):
    def __init__(self, data_path="./data", batch_size=64, num_workers=0):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']  #Classes of the dataset

    def setup(self, stage=None):
        df = pd.read_csv(f"{self.data_path}/ptbxl_database.csv", index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        agg_df = pd.read_csv(f"{self.data_path}/scp_statements.csv", index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1] 
        
        def aggregate_diagnostic(y_dict):
            tmp = []
            for key in y_dict.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            # Rimuoviamo i duplicati e i valori vuoti (nan)
            return list(set([c for c in tmp if str(c) != 'nan']))

        # Applichiamo la traduzione
        df['diagnostic_labels'] = df.scp_codes.apply(aggregate_diagnostic)
        
        # Dataset division (as indication of the Dataset) (Fold 1-8 Train, 9 Val, 10 Test)
        train_df = df[df.strat_fold <= 8]
        val_df = df[df.strat_fold == 9]
        test_df = df[df.strat_fold == 10]
        
        #Inizializing workers for each dataset
        self.train_dataset = PTBXLDataset(train_df, self.data_path, self.classes)
        self.val_dataset = PTBXLDataset(val_df, self.data_path, self.classes)
        self.test_dataset = PTBXLDataset(test_df, self.data_path, self.classes)

    #Functions for DataLoader for each dataset
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, num_workers=self.num_workers,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=self.num_workers,persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=self.num_workers,persistent_workers=True)



def plot_loss_curves(log_dir):      #Function to plot training and validation loss curves 
    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"File metrics.csv non trovato in {log_dir}")
        return
        
    df = pd.read_csv(metrics_path)
    
    if 'epoch' in df.columns:
        df_grouped = df.groupby('epoch').mean()
        
        plt.figure(figsize=(10, 6))
        
        if 'train_loss' in df_grouped.columns:
            plt.plot(df_grouped.index, df_grouped['train_loss'], label='Train Loss', marker='o', linewidth=2)
            
        if 'val_loss' in df_grouped.columns:
            plt.plot(df_grouped.index, df_grouped['val_loss'], label='Validation Loss', marker='s', linewidth=2)
            
        plt.title('Loss curves for training and validation(Loss)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (Errore)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'loss_curve.png'), dpi=300)
        print("-> Salvato: loss_curve.png")
        plt.show()


def plot_test_metrics(lightning_model, class_names, log_dir):   #Function to plot test metrics (AUROC, F1 score, ROC curve) and print the final report
    print("\n" + "="*40)
    print("      FINAL TEST METRICS REPORT (AUROC, F1 Score, ROC Curve)      ")
    print("="*40)
    
    if lightning_model.final_auroc_saved is not None:
        print(f"AUROC Macro-Average:  {lightning_model.final_auroc_saved:.4f}")
        print(f"F1-Score Macro-Average: {lightning_model.final_f1_saved:.4f}\n")
    else:
        print("Errore: Metriche non salvate correttamente in memoria.")
        return

    fpr, tpr, thresholds = lightning_model.roc_data_saved

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        plt.plot(fpr[i].cpu().numpy(), tpr[i].cpu().numpy(), lw=2, label=f'{class_names[i]}')
        
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC curves for multi-class ECG classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'roc_curve.png'), dpi=300)
    print("-> Saved: roc_curve.png")
    plt.show()



def plot_xai_examples(model, datamodule, class_names, log_dir, num_examples=3):
    # Creating a comprehensive XAI report with Vanilla Saliency for both successful pathological cases and error cases, with detailed titles and saving the images in the log directory.
    print("\n--- Generating Comprehensive XAI Report ---")
    
    model.eval()
    device = next(model.parameters()).device
    
    success_patho_indices = []
    error_indices = []
    
    loader = datamodule.test_dataloader()
    
    for batch in loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True 
        
        logits = model(inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        for i in range(len(labels)):
            is_correct = torch.equal(preds[i], labels[i])
            is_pathological = torch.sum(labels[i][1:]) > 0 
            
            if is_correct and is_pathological and len(success_patho_indices) < num_examples:
                success_patho_indices.append((inputs[i:i+1].detach().clone(), labels[i].cpu(), probs[i].detach().cpu(), "Success"))
                
            if not is_correct and len(error_indices) < num_examples:
                error_indices.append((inputs[i:i+1].detach().clone(), labels[i].cpu(), probs[i].detach().cpu(), "Error"))
                
        if len(success_patho_indices) >= num_examples and len(error_indices) >= num_examples:
            break

    def draw_single_xai(input_tensor, label, prob, category, count):
        input_tensor.requires_grad = True
        output = model(input_tensor)
        target_class_idx = torch.argmax(prob).item()
        
        model.zero_grad()
        output[0, target_class_idx].backward()
        
        saliency = input_tensor.grad[0].abs().mean(dim=0).cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        str_true = [class_names[idx] for idx, val in enumerate(label) if val == 1]
        str_pred = [f"{class_names[idx]} ({prob[idx]:.2f})" for idx, p in enumerate(prob) if p > 0.5]
        
        if not str_true: str_true = ["Nessuna"]
        if not str_pred: str_pred = ["Nessuna"]
        
        true_text = ", ".join(str_true)
        pred_text = ", ".join(str_pred)
        
        display_category = "Success case" if category == "Success" else "Error case"
        
        title = f"{display_category} #{count+1}. Model prediction: {pred_text} | Real: {true_text}\n(Area Rossa = Focus della rete su: {class_names[target_class_idx]})"

        ecg_data = input_tensor[0].detach().cpu().numpy()
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        fig, axes = plt.subplots(12, 1, figsize=(12, 18), sharex=True)
        time_axis = np.arange(ecg_data.shape[1])
        
        for j in range(12):
            axes[j].plot(time_axis, ecg_data[j], color='black', linewidth=0.8)
            axes[j].imshow(saliency[np.newaxis, :], cmap='Reds', aspect='auto', alpha=0.4, 
                           extent=[0, ecg_data.shape[1], ecg_data[j].min(), ecg_data[j].max()])
            axes[j].set_ylabel(lead_names[j], rotation=0, labelpad=20, va='center', fontweight='bold')
            axes[j].grid(alpha=0.3)
        
        plt.tight_layout()
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
        
        filename = f"xai_{category}_{count+1}.png"
        plt.savefig(os.path.join(log_dir, filename), bbox_inches='tight', dpi=300)
        print(f"-> Saved: {filename}")
        plt.close()

    for i, data in enumerate(success_patho_indices):
        draw_single_xai(data[0], data[1], data[2], data[3], i)
        
    for i, data in enumerate(error_indices):
        draw_single_xai(data[0], data[1], data[2], data[3], i)
