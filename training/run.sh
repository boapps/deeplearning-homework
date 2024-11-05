python copy_image_mask.py 
python train_cnn.py && python train_cnn_v2.py
WANDB_DISABLED=true python train_vit.py
python test_cnn.py && python test_cnn_v2.py
WANDB_DISABLED=true python test_vit.py
