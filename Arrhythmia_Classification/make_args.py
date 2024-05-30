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
        self.z_score_norm_rescale = self.config['z_score_norm_rescale']
        self.min_max_norm_rescale = self.config['min_max_norm_rescale']

        self.in_channels = self.config['in_channels']
        self.num_classes = self.config['num_classes']

        self.lr = self.config['lr']
        self.weight_decay = self.config['weight_decay']
        self.num_epochs = self.config['num_epochs']
        self.monitor = self.config['monitor']
        self.model_save_path = self.config['model_save_path']

        self.dist_proc_port = self.config['dist_proc_port']