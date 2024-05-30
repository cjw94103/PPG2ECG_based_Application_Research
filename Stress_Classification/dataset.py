import torch
import sklearn.preprocessing as skp
import numpy as np
import cv2

from torch.utils import data
from scipy.interpolate import splrep, splev
from biosppy.signals import tools as tools

def shuffle_filepath(paths):
    idx_list = [i for i in range(len(paths))]
    np.random.shuffle(idx_list)
    paths_shuffle = [paths[idx_list[i]] for i in range(len(paths))]
    
    return paths_shuffle

def interp_spline(ecg, step=1, k=3):
    x_new = np.arange(0, ecg.shape[0], ecg.shape[0]/step)
    interp_spline_method = splrep(np.arange(0, ecg.shape[0], 1), ecg, k=k)
    return splev(x_new, interp_spline_method)

def filter_ecg(signal, sampling_rate):
    
    signal = np.array(signal)
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=[3, 45],
                                  sampling_rate=sampling_rate)
    
    return filtered

def filter_ppg(signal, sampling_rate):
    
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='butter',
                                  band='bandpass',
                                  order=4, #3
                                  frequency=[0.5, 8], #[0.5, 8]
                                  sampling_rate=sampling_rate)

    return filtered

class PPG_Dataset(data.Dataset):
    def __init__(self, filepaths, sampling_rate=128., min_max_norm=True, z_score_norm=True, interp='spline'):
        
#         self.filepaths = shuffle_filepath(filepaths)
        self.filepaths = filepaths
        self.sampling_rate = sampling_rate
        self.min_max_norm = min_max_norm
        self.z_score_norm = z_score_norm
        self.interp = interp ## spline or cv2_linear
 
    def __len__(self):
        return len(self.filepaths)
    
    def prepare_data(self, index):
        data_dict_ppg = np.load(self.filepaths[index], allow_pickle=True).item()
        
        original_ppg_fs = data_dict_ppg['PPG']['sig_fs']
        original_ppg_len = data_dict_ppg['PPG']['sig_len']
        original_ppg = data_dict_ppg['PPG']['sig']
        label = data_dict_ppg['label']
        
        # interpolation
        if self.interp == 'spline':
            ppg_sig_seconds = len(original_ppg) // original_ppg_fs
            ppg = interp_spline(original_ppg, step=self.sampling_rate*ppg_sig_seconds, k=5)
            
        elif self.interp == 'cv2_linear':
            ppg_sig_seconds = len(original_ppg) // original_ppg_fs
            ppg = cv2.resize(original_ppg, (1, self.sampling_rate*ppg_sig_seconds), interpolation=cv2.INTER_LINEAR).flatten()
            
        elif self.interp == None:
            ppg = original_ppg
            self.sampling_rate = original_ppg_fs
            
        # filtering
        ppg = filter_ppg(ppg, self.sampling_rate)
          
        # reshape
        ppg_seg = np.reshape(ppg, (1, -1))
        
        # z-score norm
        if self.z_score_norm == True:
            ppg_seg = (ppg_seg-ppg_seg.mean()) / (ppg_seg.std() + 1e-17)
            
        # self min-max normalize
        if self.min_max_norm == True:
            ppg_seg = skp.minmax_scale(ppg_seg, (-1, 1), axis=1)
            
        # array to torch tensor
        ppg_seg = torch.from_numpy(ppg_seg).type(torch.FloatTensor)
        
        # label 작업
#         label = torch.from_numpy(label).type(torch.LongTensor)
            
        return ppg_seg, label
    
    def __getitem__(self, index):
        return self.prepare_data(index)
    