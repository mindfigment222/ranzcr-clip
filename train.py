import time
import copy
import torch


def train_model(model, criterion, optimizer, scheduler, num_epochs, t_gen, v_gen, t_size, v_size, device):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc= 0
    
    # Loop over epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Training
        train_running_loss = 0
        train_running_corrects = 0
        model.train()
        for inputs, labels in t_gen:

            print(inputs)
            print(labels)
            # Transfer to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(inputs)
            _, preds = torch.ge(outputs, 0.5).int()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * inputs.size(0)
            train_running_corrects += torch.sum(preds == labels.data)

        # Validation
        val_running_loss = 0
        val_running_corrects = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in v_gen:
                # Transfer to GPU
                inputs = inputs.to(device)
                outputs = outputs.to(device)

                # Model computations
                outputs = model(inputs)
                _, preds = torch.ge(outputs, 0.5).int()
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = train_running_loss / t_size
        epoch_train_acc = train_running_corrects.double() / t_size
        
        epoch_val_loss = val_running_loss / v_size
        epoch_val_acc= val_running_corrects.double() / v_size
        
        print('train => loss: {:.4f} acc: {:.4f}'.format(epoch_train_loss, epoch_train_acc))
        print('val   => loss: {:.4f} acc: {:.4f}'.format(epoch_val_loss, epoch_val_acc))
        print()
        
        # deep copy the model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        scheduler.step()
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model