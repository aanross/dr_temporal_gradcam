import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score
from sklearn.metrics import cohen_kappa_score
import cv2
import numpy as np

from src.models import get_model
from src.visualizations import compute_iou, get_binary_cam_mask, get_cam_extractor

class DRLightningModule(pl.LightningModule):
    def __init__(self, model_name="resnet50_lstm", num_classes=5, lr=1e-4, cam_method="gradcam"):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name, num_classes)
        self.lr = lr
        self.cam_method = cam_method
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Metrics setup
        self.dx_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.dx_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.prog_auc = AUROC(task="binary")
        
        # To store predictions per epoch
        self.val_dx_preds = []
        self.val_dx_targets = []
        self.val_prog_preds = []
        self.val_prog_targets = []
        
        # Hook point for torchcam (for simplicity we attach to the model's innermost CNN block if possible)
        # Note: Depending on the complex model, finding the target layer programmatically can be tricky.
        # TorchCAM auto-finder usually grabs the last conv layer if passed None.
        self.cam_extractor = None 
        # We'll instantiate the extractor outside the training loop (e.g., during val/test) to save memory 

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx, mode="train"):
        x = batch['pixel_values']
        dx_target = batch['final_diagnosis']
        prog_target = batch['progression'].unsqueeze(1)
        
        dx_logits, prog_logits = self(x)
        
        loss_dx = self.ce_loss(dx_logits, dx_target)
        loss_prog = self.bce_loss(prog_logits, prog_target)
        
        total_loss = loss_dx + loss_prog
        
        self.log(f'{mode}_loss', total_loss, prog_bar=True)
        self.log(f'{mode}_dx_loss', loss_dx)
        self.log(f'{mode}_prog_loss', loss_prog)
        
        # Log metrics
        if mode == "val":
            # For Accuracy and F1, we need softmax probes
            dx_preds = torch.argmax(dx_logits, dim=1)
            self.dx_acc(dx_preds, dx_target)
            self.dx_f1(dx_preds, dx_target)
            
            # For AUC, we need probabilities
            prog_probs = torch.sigmoid(prog_logits)
            self.prog_auc(prog_probs, prog_target.squeeze())
            
            self.log('val_dx_acc', self.dx_acc, on_step=False, on_epoch=True)
            self.log('val_dx_f1', self.dx_f1, on_step=False, on_epoch=True)
            self.log('val_prog_auc', self.prog_auc, on_step=False, on_epoch=True)
            
            self.val_dx_preds.append(dx_preds.detach().cpu())
            self.val_dx_targets.append(dx_target.detach().cpu())
            self.val_prog_preds.append(prog_probs.detach().cpu())
            self.val_prog_targets.append(prog_target.detach().cpu())
            
            # --- EVALUATE GRADCAM IOU ---
            # We'll only do it for the first batch to not slow down validation too much
            if batch_idx == 0:
                self.evaluate_gradcam(batch)
                
        return total_loss

    def evaluate_gradcam(self, batch):
        """Runs the chosen CAM method and calculates Lesion IoU."""
        target_model = self.model.backbone if hasattr(self.model, 'backbone') else self.model
        
        # Instantiate extractor locally and automatically clean up hooks when done
        cam_extractor = get_cam_extractor(target_model, self.cam_method, target_layer=None)
        if cam_extractor is None:
            return # Skip if unable to extract CAM
            
        x = batch['pixel_values']            # (B, T, C, H, W)
        masks = batch['lesion_masks']        # (B, T, 1, H, W)
        B, T, C, H, W = x.shape
        
        total_iou = 0
        valid_frames = 0
        
        # We need to reshape x to pass it through if it's not the baseline ResNet
        # Or we can just evaluate GradCAM on individual frames through the backbone
        # This is a safe fallback to compute spatial GradCAM IoU cleanly for any architecture 
        target_model = self.model.backbone if hasattr(self.model, 'backbone') else self.model
        
        # Pass each frame independently for CAM calculation
        with torch.enable_grad():
            x_flat = x.view(B * T, C, H, W).requires_grad_(True)
            out = target_model(x_flat)
            
            # The target class is just the argmax to see what the feature extractor focused on
            if out.dim() > 1 and out.shape[1] >= 5: # If it outputs logits natively
                class_idx = out.argmax(dim=1).tolist()
            else: # If it just outputs features (e.g., our custom models strip the FC), we can't do class-specific CAM
                # We must pass through the full path if we want class CAM.
                # Instead, we will simulate a generic CAM extraction by just using the sum of features
                class_idx = None 
                
            try:
                cams = cam_extractor(class_idx, out)
                if isinstance(cams, list):
                    cams = cams[0] # Take the first layer's CAM if multiple are returned
                    
                # Resize CAMs to image size
                cams = torch.nn.functional.interpolate(cams.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False)
                
                cams_np = cams.squeeze(1).detach().cpu().numpy()
                masks_np = masks.view(B * T, H, W).detach().cpu().numpy()
                
                for i in range(len(cams_np)):
                    cam_norm = (cams_np[i] - cams_np[i].min()) / (cams_np[i].max() - cams_np[i].min() + 1e-8)
                    cam_bin = get_binary_cam_mask(cam_norm, threshold=0.3)
                    mask_bin = (masks_np[i] > 0.5).astype(np.uint8)
                    
                    if mask_bin.sum() > 0: # Only compute IoU if there's a lesion
                        iou = compute_iou(cam_bin, mask_bin)
                        total_iou += iou
                        valid_frames += 1
                        
            except Exception as e:
                # print(f"CAM Execution failed: {e}")
                pass
            finally:
                # Clean up hooks to prevent crashing subsequent validation forward passes
                if hasattr(cam_extractor, 'remove_hooks'):
                    cam_extractor.remove_hooks()
            
        if valid_frames > 0:
            avg_iou = total_iou / valid_frames
            self.log('val_lesion_iou', avg_iou)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")
        
    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, mode="val")

    def on_validation_epoch_end(self):
        # Calculate QWK
        if len(self.val_dx_preds) > 0:
            preds = torch.cat(self.val_dx_preds).numpy()
            targets = torch.cat(self.val_dx_targets).numpy()
            
            # Using scikit-learn for QWK
            qwk = cohen_kappa_score(targets, preds, weights='quadratic')
            self.log('val_dx_qwk', qwk)
            
            self.val_dx_preds.clear()
            self.val_dx_targets.clear()
            self.val_prog_preds.clear()
            self.val_prog_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
