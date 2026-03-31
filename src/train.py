import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

from src.dataset import DRDataModule
from src.lightning_module import DRLightningModule
from src.visualizations import plot_roc_curves, plot_radar_chart, plot_cam_gallery, compute_delong_test

def main():
    parser = argparse.ArgumentParser(description="Time-aware DR Detection Training")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for APTOS dataset")
    parser.add_argument("--model_type", type=str, default="resnet50_lstm", 
                        choices=["resnet_baseline", "resnet50_lstm", "efficientnet_bilstm", 
                                 "vit_temporal", "timesformer", "convlstm"])
    parser.add_argument("--cam_method", type=str, default="gradcam", 
                        choices=["gradcam", "gradcam++", "scorecam", "layercam"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate_plots", action="store_true", help="Generate comparative plots with dummy logic at end")
    
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    # K-Fold Cross Validation Loop
    fold_results = []
    
    for fold in range(args.num_folds):
        print(f"\\n{'='*20} Starting Fold {fold+1}/{args.num_folds} {'='*20}")
        
        # 1. Init Data
        datamodule = DRDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            fold=fold,
            num_folds=args.num_folds,
            seed=args.seed,
            num_patients=3662
        )
        
        # 2. Init Model
        model = DRLightningModule(
            model_name=args.model_type,
            cam_method=args.cam_method,
            lr=1e-4
        )
        
        # 3. Setup Logger and Callbacks
        run_name = f"{args.model_type}_{args.cam_method}_fold{fold}"
        wandb_logger = WandbLogger(project="DR_Temporal_GradCAM", name=run_name)
        
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"checkpoints/{run_name}",
            filename="best-checkpoint",
            save_top_k=1,
            mode="min"
        )
        early_stop_cb = EarlyStopping(monitor="val_loss", patience=5)
        
        # 4. Train
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_cb, early_stop_cb],
            devices="auto",
            accelerator="auto"
        )
        
        trainer.fit(model, datamodule=datamodule)
        
        # 5. Extract a batch for CAM visual saving
        # Fetch one batch from validation to generate CAM gallery
        datamodule.setup()
        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))
        
        # Force a CAM eval to get the gallery
        model.eval()
        with torch.no_grad():
            _ = model(batch['pixel_values'])
            
        # Optional: Save CAM plot locally for the first item in batch
        # Assuming the extractor ran and saved heatmaps internally during validation 
        # For a full project, we would explicitly extract and pass to plot_cam_gallery here.
        # This is delegated since the visualization logic exists in `visualizations.py`
        print(f"Fold {fold} finished. Best model saved at: {checkpoint_cb.best_model_path}")
        wandb_logger.experiment.finish()

    if args.generate_plots:
        # Generate dummy/placeholder visualizations to demonstrate functionality
        print("Generating Comparative Study Visualizations...")
        os.makedirs("results", exist_ok=True)
        
        # 1. Dummy ROC
        models = ["ResNet50", "ResNet50+LSTM", "EfficientNet+BiLSTM", "ViT+Temporal", "TimeSformer", "ConvLSTM"]
        y_true = np.random.randint(0, 2, 100)
        y_preds = [np.clip(y_true * 0.5 + np.random.rand(100) * 0.5, 0, 1) for _ in models] # simulate predictions
        plot_roc_curves([y_true]*6, y_preds, models, save_path="results/roc_comparison.png")
        
        # 2. Dummy Radar Chart
        radar_metrics = {
            m: [np.random.uniform(0.6, 0.95) for _ in range(4)] for m in models
        }
        plot_radar_chart(radar_metrics, save_path="results/radar_comparison.png")
        
        # 3. DeLong test table
        cdd_p_val = compute_delong_test(y_preds[1], y_preds[2], y_true)
        with open("results/delong_stats.txt", "w") as f:
            f.write("DeLong/Wilcoxon Test P-values (vs Baseline):\n")
            f.write(f"ResNet50+LSTM vs EfficientNet+BiLSTM: p = {cdd_p_val:.4f}\n")
            
        print("Outputs saved to results/ directory.")

if __name__ == "__main__":
    main()
