class Config:
    PATH = 'C:/Adrian/Egyetem-Msc/Deep Learning Python es LUA alapon/Project/RUN_dir/'
    WEIGHTS_PATH = f'{PATH}weights/'


class Parameters:
    num_classes = 50
    num_epochs = 50
    learning_rate = 1e-3
    min_lr = 0
    reduce_lr_factor = 0.2
    reduce_lr_patience = 1
    early_stop_patience = 3
    momentum = 0.9
    verbose = 0
    weight_decay = 0.1
    image_size = 64 # 256 was, but there was a memory allocation error, and because of that we could not fit the ViT (memory issues)
    patch_size = 16
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64 # 64 ORIGINALLLY
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    transformer_layers = 8
    mlp_head_units = [2048, 1024]
