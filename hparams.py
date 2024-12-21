import tensorflow as tf
from text import symbols


class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def create_hparams(hparams_string=None, verbose=False):
    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=100,
        iters_per_checkpoint=200,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=True,
        training_files='/content/filelist.txt',
        validation_files='/content/filelist.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=2,  # Reduce convolutions
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.2,  # Increase dropout
        p_decoder_dropout=0.2,  # Increase dropout

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-4,  # Lower learning rate
        weight_decay=1e-6,
        grad_clip_thresh=0.5,  # Lower grad clip threshold
        batch_size=16,  # Smaller batch size for small dataset
        mask_padding=True  # set model's padded outputs to padded values
    )
    return hparams

