import numpy as np
import torch

from utils import AverageMeter
from sklearn.metrics import classification_report
from tqdm import tqdm
import sklearn.preprocessing as skp

def train_PPG_model(model, train_dataloader, val_dataloader, z_score_norm, min_max_norm,
                    num_epochs, optimizer, base_lr, monitor, model_save_path, multi_gpu_flag, DEVICE):
    
    def save_history(train_history, val_history, save_path):
        history = {}
        history['train_total_loss'] = np.asarray(train_history[0])
        history['val_total_loss'] = np.asarray(val_history[0])
        history['val_avg_f1'] = np.asarray(val_history[1])

        np.save(save_path, history) 
        
    def lr_cosine_decay(base_learning_rate, global_step, decay_steps, alpha=0):
        """
        Params
            - learning_rate : Base Learning Rate
            - global_step : Current Step in Train Pipeline
            - decay_steps : Total Decay Steps in Learning Rate
            - alpha : Learning Scaled Coefficient
        """
        global_step = min(global_step, decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        decayed_learning_rate = base_learning_rate * decayed

        return decayed_learning_rate
    
    def normalize_data(data, z_score_norm=True, min_max_norm=False):
        norm_data = []
        data = data.data.cpu().numpy()
        
        for i in range(len(data)):
            target_data = data[i].copy()
            if z_score_norm:
                target_data = (target_data - target_data.mean()) / (target_data.std() + 1e-17)
            if min_max_norm:
                target_data = skp.minmax_scale(target_data, (-1, 1), axis=1)
            norm_data.append(target_data)
            
        return torch.from_numpy(np.array(norm_data)).type(torch.FloatTensor)

    criterion = torch.nn.CrossEntropyLoss()
    start_epoch = 0
    total_iter = len(train_dataloader) * num_epochs
    global_step = 0
    minimum_val_loss = float("inf")
    minimum_val_perf = 0
    
    train_losses_avg = []
    val_losses_avg = []
    epoch_val_perf = []

    for epoch in range(start_epoch, num_epochs):
        train_losses = AverageMeter()
        val_losses = AverageMeter()

        # train
        model.train()
        train_t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for iter, train_data in train_t:
            # data extraction
            ppg, label = train_data
            ppg = ppg.to(DEVICE)
            label = label.type(torch.LongTensor).to(DEVICE)     

            if z_score_norm == True and min_max_norm == False:
                ppg = normalize_data(ppg, z_score_norm=True, min_max_norm=False)
                
            elif z_score_norm == False and min_max_norm == True:
                ppg = normalize_data(ppg, z_score_norm=False, min_max_norm=True)
                
            elif z_score_norm == True and min_max_norm == True:
                ppg = normalize_data(ppg, z_score_norm=True, min_max_norm=True)
                
            elif z_score_norm == False and min_max_norm == False:
                ppg = normalize_data(ppg, z_score_norm=False, min_max_norm=False)

            input_data = ppg.to(DEVICE)

            # calculate loss
            y_pred = model(input_data)    
            train_loss = criterion(y_pred, label)
            train_losses.update(train_loss.item())

            # weight update
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # adjust optimizer lr
            global_step += 1
            adjust_lr = lr_cosine_decay(base_lr, global_step, total_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjust_lr

            # visualize loss per iterations
            train_t.set_postfix_str("train_loss : " + str(round(train_loss.item(), 3)))

        # loss recording
        train_losses_avg.append(train_losses.avg)

        # validation
        model.eval()
        val_t = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

        total_y_val = []
        total_y_pred_val = []

        with torch.no_grad():
            for iter, val_data in val_t:
                # data extraction
                ppg, label = val_data
                ppg = ppg.to(DEVICE)
                label = label.type(torch.LongTensor).to(DEVICE)
                
                if z_score_norm == True and min_max_norm == False:
                    ppg = normalize_data(ppg, z_score_norm=True, min_max_norm=False)

                elif z_score_norm == False and min_max_norm == True:
                    ppg = normalize_data(ppg, z_score_norm=False, min_max_norm=True)

                elif z_score_norm == True and min_max_norm == True:
                    ppg = normalize_data(ppg, z_score_norm=True, min_max_norm=True)

                elif z_score_norm == False and min_max_norm == False:
                    ppg = normalize_data(ppg, z_score_norm=False, min_max_norm=False)

                input_data = ppg.to(DEVICE)

                y_pred = model(input_data)
                val_loss = criterion(y_pred, label)
                val_losses.update(val_loss.item())

                # for avg-f1
                total_y_val.extend(label.data.cpu().numpy())
                total_y_pred_val.extend(y_pred.data.cpu().numpy())

                val_t.set_postfix_str("val_loss : " + str(round(val_loss.item(), 3)))

        # loss recording
        val_losses_avg.append(val_losses.avg)

        # calculate avg-f1
        total_y_val = np.array(total_y_val)
        total_y_pred_val = np.argmax(np.array(total_y_pred_val), -1)
        val_average_f1 = classification_report(total_y_val, total_y_pred_val, digits=4, output_dict=True)['macro avg']['f1-score']

        epoch_val_perf.append(val_average_f1)

        # print train loss, val loss
        print("Epoch : {}   Train Loss : {}  Val Loss : {}".format(epoch+1, round(train_losses.avg, 4), round(val_losses.avg, 4)))

        # save best model
        if monitor == 'loss':
            val_avg_loss = val_losses.avg
            if val_avg_loss<minimum_val_loss:
                print('improve val_loss!! so model save {} -> {}'.format(minimum_val_loss, val_avg_loss))
                minimum_val_loss = val_avg_loss
                if multi_gpu_flag == True:
                    torch.save(model.module.state_dict(), model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)

        elif monitor == 'average_f1':
            if val_average_f1>minimum_val_perf:
                print('improve val_perf!! so model save {} -> {}'.format(minimum_val_perf, val_average_f1))
                minimum_val_perf = val_average_f1
                if multi_gpu_flag == True:
                    torch.save(model.module.state_dict(), model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)

        # save history
        save_history([train_losses_avg], [val_losses_avg, epoch_val_perf], save_path = model_save_path.replace('.pth', '.npy'))

def train_PPGECG_model(model, G_AB, train_dataloader, val_dataloader, z_score_norm, min_max_norm,
                    num_epochs, optimizer, base_lr, monitor, model_save_path, multi_gpu_flag, DEVICE):
    
    def save_history(train_history, val_history, save_path):
        history = {}
        history['train_total_loss'] = np.asarray(train_history[0])
        history['val_total_loss'] = np.asarray(val_history[0])
        history['val_avg_f1'] = np.asarray(val_history[1])

        np.save(save_path, history) 
        
    def lr_cosine_decay(base_learning_rate, global_step, decay_steps, alpha=0):
        """
        Params
            - learning_rate : Base Learning Rate
            - global_step : Current Step in Train Pipeline
            - decay_steps : Total Decay Steps in Learning Rate
            - alpha : Learning Scaled Coefficient
        """
        global_step = min(global_step, decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        decayed_learning_rate = base_learning_rate * decayed

        return decayed_learning_rate

    def make_syn_ecg(G_AB, input_ppg):
        with torch.no_grad():
            syn_ecg = G_AB(input_ppg)

        return syn_ecg
        
    def normalize_data(data, z_score_norm=True, min_max_norm=False):
        norm_data = []
        data = data.data.cpu().numpy()
        
        for i in range(len(data)):
            target_data = data[i].copy()
            if z_score_norm:
                target_data = (target_data - target_data.mean()) / (target_data.std() + 1e-17)
            if min_max_norm:
                target_data = skp.minmax_scale(target_data, (-1, 1), axis=1)
            norm_data.append(target_data)
            
        return torch.from_numpy(np.array(norm_data)).type(torch.FloatTensor)

    criterion = torch.nn.CrossEntropyLoss()
    start_epoch = 0
    total_iter = len(train_dataloader) * num_epochs
    global_step = 0
    minimum_val_loss = float("inf")
    minimum_val_perf = 0

    if G_AB != None:
        G_AB = G_AB.to(DEVICE)
        G_AB = G_AB.eval()

    train_losses_avg = []
    val_losses_avg = []
    epoch_val_perf = []

    for epoch in range(start_epoch, num_epochs):
        train_losses = AverageMeter()
        val_losses = AverageMeter()

        # train
        model.train()
        train_t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for iter, train_data in train_t:
            # data extraction
            ppg, label = train_data
            ppg = ppg.to(DEVICE)
            label = label.type(torch.LongTensor).to(DEVICE)
            ecg = make_syn_ecg(G_AB, ppg)            

            if z_score_norm == True and min_max_norm == False:
                ppg = normalize_data(ppg, z_score_norm=True, min_max_norm=False)
                ecg = normalize_data(ecg, z_score_norm=True, min_max_norm=False)
                
            elif z_score_norm == False and min_max_norm == True:
                ppg = normalize_data(ppg, z_score_norm=False, min_max_norm=True)
                ecg = normalize_data(ecg, z_score_norm=False, min_max_norm=True)
                
            elif z_score_norm == True and min_max_norm == True:
                ppg = normalize_data(ppg, z_score_norm=True, min_max_norm=True)
                ecg = normalize_data(ecg, z_score_norm=True, min_max_norm=True)
                
            elif z_score_norm == False and min_max_norm == False:
                ppg = normalize_data(ppg, z_score_norm=False, min_max_norm=False)
                ecg = normalize_data(ecg, z_score_norm=False, min_max_norm=False)

            input_data = torch.cat((ppg, ecg), 1).to(DEVICE)

            # calculate loss
            y_pred = model(input_data)    
            train_loss = criterion(y_pred, label)
            train_losses.update(train_loss.item())

            # weight update
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # adjust optimizer lr
            global_step += 1
            adjust_lr = lr_cosine_decay(base_lr, global_step, total_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjust_lr

            # visualize loss per iterations
            train_t.set_postfix_str("train_loss : " + str(round(train_loss.item(), 3)))

        # loss recording
        train_losses_avg.append(train_losses.avg)

        # validation
        model.eval()
        val_t = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

        total_y_val = []
        total_y_pred_val = []

        with torch.no_grad():
            for iter, val_data in val_t:
                # data extraction
                ppg, label = val_data
                ppg = ppg.to(DEVICE)
                label = label.type(torch.LongTensor).to(DEVICE)
                ecg = make_syn_ecg(G_AB, ppg)
                
                if z_score_norm == True and min_max_norm == False:
                    ppg = normalize_data(ppg, z_score_norm=True, min_max_norm=False)
                    ecg = normalize_data(ecg, z_score_norm=True, min_max_norm=False)

                elif z_score_norm == False and min_max_norm == True:
                    ppg = normalize_data(ppg, z_score_norm=False, min_max_norm=True)
                    ecg = normalize_data(ecg, z_score_norm=False, min_max_norm=True)

                elif z_score_norm == True and min_max_norm == True:
                    ppg = normalize_data(ppg, z_score_norm=True, min_max_norm=True)
                    ecg = normalize_data(ecg, z_score_norm=True, min_max_norm=True)

                elif z_score_norm == False and min_max_norm == False:
                    ppg = normalize_data(ppg, z_score_norm=False, min_max_norm=False)
                    ecg = normalize_data(ecg, z_score_norm=False, min_max_norm=False)

                input_data = torch.cat((ppg, ecg), 1).to(DEVICE)

                y_pred = model(input_data)
                val_loss = criterion(y_pred, label)
                val_losses.update(val_loss.item())

                # for avg-f1
                total_y_val.extend(label.data.cpu().numpy())
                total_y_pred_val.extend(y_pred.data.cpu().numpy())

                val_t.set_postfix_str("val_loss : " + str(round(val_loss.item(), 3)))

        # loss recording
        val_losses_avg.append(val_losses.avg)

        # calculate avg-f1
        total_y_val = np.array(total_y_val)
        total_y_pred_val = np.argmax(np.array(total_y_pred_val), -1)
        val_average_f1 = classification_report(total_y_val, total_y_pred_val, digits=4, output_dict=True)['macro avg']['f1-score']

        epoch_val_perf.append(val_average_f1)

        # print train loss, val loss
        print("Epoch : {}   Train Loss : {}  Val Loss : {}".format(epoch+1, round(train_losses.avg, 4), round(val_losses.avg, 4)))

        # save best model
        if monitor == 'loss':
            val_avg_loss = val_losses.avg
            if val_avg_loss<minimum_val_loss:
                print('improve val_loss!! so model save {} -> {}'.format(minimum_val_loss, val_avg_loss))
                minimum_val_loss = val_avg_loss
                if multi_gpu_flag == True:
                    torch.save(model.module.state_dict(), model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)

        elif monitor == 'average_f1':
            if val_average_f1>minimum_val_perf:
                print('improve val_perf!! so model save {} -> {}'.format(minimum_val_perf, val_average_f1))
                minimum_val_perf = val_average_f1
                if multi_gpu_flag == True:
                    torch.save(model.module.state_dict(), model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)

        # save history
        save_history([train_losses_avg], [val_losses_avg, epoch_val_perf], save_path = model_save_path.replace('.pth', '.npy'))