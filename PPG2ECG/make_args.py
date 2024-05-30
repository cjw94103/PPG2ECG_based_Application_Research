from utils import load_json_file

class Args:
    def __init__(self, config_path):
        self.config = load_json_file(config_path)
        self.partition_path = self.config['partition_path']
        self.num_workers = self.config['num_workers']
        self.batch_size = self.config['batch_size']

        self.target_sampling_rate = self.config['target_sampling_rate']
        self.min_max_norm = self.config['min_max_norm']
        self.z_score_norm = self.config['z_score_norm']
        self.sig_time_len = self.config['sig_time_len']
        self.interp_method = self.config['interp_method']

        self.n_residual_blocks = self.config['n_residual_blocks']

        self.lambda_cyc = self.config['lambda_cyc']
        self.lr = self.config['lr']
        self.b1 = self.config['b1']
        self.b2 = self.config['b2']
        self.lr_decay_epoch = self.config['lr_decay_epoch']
        self.num_epochs = self.config['num_epochs']
        self.save_per_epochs = self.config['save_per_epochs']
        self.model_save_path = self.config['model_save_path']

        self.dist_proc_port = self.config['dist_proc_port']