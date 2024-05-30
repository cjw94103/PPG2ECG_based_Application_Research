import torch
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
    
def train_CycleGAN(G_AB, G_BA, D_A, D_B, train_dataloader, lambda_cyc, D_output_shape, num_epochs, early_stop_patience,
                   optimizer_G, optimizer_D_A, optimizer_D_B, lr_decay_epoch, base_lr, monitor, save_per_epochs, 
                   model_save_path, multi_gpu_flag, DEVICE):
    
    def save_history(D_loss_list, G_loss_list, save_path):
        history = {}

        history['D_loss'] = D_loss_list
        history['G_loss'] = G_loss_list

        np.save(save_path, history)
        
    start_epoch = 0
    total_iter = len(train_dataloader) * num_epochs
    
    D_loss_list = []
    G_loss_list = []
    
    # get optimizer scheduler
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(num_epochs, 0, lr_decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(num_epochs, 0, lr_decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(num_epochs, 0, lr_decay_epoch).step
    )
    
    # define loss instance
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    
    for epoch in range(start_epoch, num_epochs):
        train_t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        # train
        for i, batch in train_t:
            # set model input
            real_A, real_B = batch['ppg'].to(DEVICE), batch['ecg'].to(DEVICE)

            # Adversarial ground truths
            valid = Variable(torch.Tensor(np.ones(D_output_shape)), 
                             requires_grad=False).to(DEVICE)
            fake = Variable(torch.Tensor(np.zeros(D_output_shape)), 
                            requires_grad=False).to(DEVICE)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad() 

            # GAN(adversarial) loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total Generator Loss
            loss_G = loss_GAN + lambda_cyc * loss_cycle

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
    #         fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            fake_A_ = fake_A
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
    #         fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            fake_B_ = fake_B
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            # Total Discriminator loss
            loss_D = (loss_D_A + loss_D_B) / 2
            
            # loss recording
            D_loss_list.append(loss_D.item())
            G_loss_list.append(loss_G.item())

            # print tqdm
            print_D_loss = round(loss_D.item(), 4)
            print_G_loss = round(loss_G.item(), 4)
            train_t.set_postfix_str("Discriminator loss : {}, Generator loss : {}".format(print_D_loss, print_G_loss))
            
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
            
        # save loss dict
        save_history(D_loss_list, G_loss_list, model_save_path.replace('.pth', '.npy'))
        
        # save model per epochs
        if save_per_epochs is not None:
            if (epoch+1) % save_per_epochs == 0:
                print("save per epochs {}".format(str(epoch+1)))
                per_epoch_save_path = model_save_path.replace(".pth", '_' + str(epoch+1) + 'Epochs.pth')

                if multi_gpu_flag == True:
                    model_dict = {}
                    model_dict['D_A'] = D_A.module.state_dict()
                    model_dict['D_B'] = D_B.module.state_dict()
                    model_dict['G_AB'] = G_AB.module.state_dict()
                    model_dict['G_BA'] = G_BA.module.state_dict()
                    torch.save(model_dict, per_epoch_save_path)
                else:
                    model_dict = {}
                    model_dict['D_A'] = D_A.state_dict()
                    model_dict['D_B'] = D_B.state_dict()
                    model_dict['G_AB'] = G_AB.state_dict()
                    model_dict['G_BA'] = G_BA.state_dict()
                    torch.save(model_dict, per_epoch_save_path)