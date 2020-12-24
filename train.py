import time
import copy
import torch
import numpy as np
from tqdm import tqdm
import os

from utils import init_logger, get_score


def train_model(model, criterion, optimizer, scheduler, num_epochs, t_gen, v_gen, save_dir, log_dir, model_dir, device):
    
    # Init logger
    logger = init_logger(log_dir)

    # Path where to save best model
    save_path = os.path.join(save_dir, 'best.pth')

    # Save start time of training to see how long it took
    train_start = time.time()
    
    # Vars for keeping track of best model auc score and epoch in which it occured
    best_auc = 0
    best_epoch = 0

    # Training and validaiton set sizes for normalizing loss
    t_size = len(t_gen.dataset)
    v_size = len(v_gen.dataset)
    
    ######################
    ## Loop over epochs ##
    ######################

    for epoch in range(num_epochs):

        # Save epoch start time to see how long it took
        epoch_start = time.time()
        
        ##############
        ## Training ##
        ##############
        
        train_running_loss = 0
        model.train()
        for inputs, labels in tqdm(t_gen):

            # Transfer to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * inputs.size(0)

        ################
        ## Validation ##
        ################

        val_running_loss = 0
        avg_auc = 0
        auc_data = []
        labels_for_auc = []
        probs_for_auc = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in v_gen:
                # Transfer to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Model computations
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=0)
                loss = criterion(outputs, labels)

                auc_data.append([labels.detach().cpu().numpy(), probs.detach().cpu().numpy()])

                # labels_for_auc.append(labels.detach().cpu().numpy())
                # probs_for_auc.append(probs.detach().cpu().numpy())
                
                val_running_loss += loss.item() * inputs.size(0)

        #############
        ## Logging ##
        #############

        epoch_train_loss = train_running_loss / t_size
        epoch_val_loss = val_running_loss / v_size
        avg_auc, aucs, avg_log, logs = get_score(np.concatenate([data[0] for data in auc_data]), np.concatenate([data[1] for data in auc_data]))
        
        epoch_duration = time.time() - epoch_start
        logger.info('Epoch {}/{} ({:.0f}m {:.0f}s)'.format(epoch + 1, num_epochs, epoch_duration // 60,  epoch_duration % 60))
        logger.info('train => loss: {:.4f}'.format(epoch_train_loss))
        logger.info('val   => loss: {:.4f} avg_auc: {:.4f} avg_log: {:.4f}'.format(epoch_val_loss, avg_auc, avg_log))
        logger.info('         aucs: {}'.format(np.round(aucs, decimals=3)))
        logger.info('         logs: {}'.format(np.round(logs, decimals=3)))
        logger.info(' ')
        
        ############
        ## Saving ##
        ############

        if avg_auc > best_auc:
            best_auc = avg_auc
            best_epoch = epoch
            torch.save({
                'model_dir': model_dir,
                'epoch' : epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'avg_auc': avg_auc,
                'aucs': aucs,
                'avg_log': avg_log,
                'logs': logs   
            }, save_path)
            
            logger.info('Saving model! New best auc: {} in epoch: {}'.format(best_auc, best_epoch))
            logger.info(' ')

        # Updating scheduler after every epoch
        scheduler.step()

        logger.info('-' * 50)
        logger.info(' ')

    ###############
    ## Final log ##
    ###############

    time_elapsed = time.time() - train_start
    logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val auc: {:4f} in epoch: {}'.format(best_auc, best_epoch))
    
    ################################
    ## Loading best model weights ##
    ################################

    checkpoint = model.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model
