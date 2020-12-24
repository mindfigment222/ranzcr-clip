from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
from sklearn.metrics import roc_auc_score, log_loss
import os
import datetime
import numpy as np
import torchvision.transforms as T
import pandas as pd


def get_test_img_uids(test_csv='/home/mszmelcz/Datasets/ranzcr-clip/sample_submission.csv'):
    submission_df = pd.read_csv(test_csv)
    img_uids = submission_df['StudyInstanceUID'].to_list()
    return img_uids, submission_df


def get_transforms(resize=256, crop_size=224):
    data_transforms = {
        'train': T.Compose([
            T.Resize((resize, resize)),
            T.CenterCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': T.Compose([
            T.Resize((resize, resize)),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    return data_transforms


def make_train_directories(model_dir):
    # Get current date for saving folder
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    # Initialise the run and dir_check to create a new folde within the current date
    run = 0
    dir_check = True
    tensorboard_path, params_path, run_path = None, None, None
    # Find the current run: the first run that doesn't exist yet
    while dir_check:
        # Construct new paths
        run_path = os.path.join('.', 'summaries', model_dir, date, 'run' + str(run))
        params_path = os.path.join(run_path, 'params')
        tensorboard_path = os.path.join(run_path, 'tensorboard')
        run += 1
        # And once a path doesn't exist yet: create new folders
        if not os.path.exists(run_path):
            os.makedirs(params_path)
            os.makedirs(tensorboard_path)
            dir_check = False
    # Return folders to new path
    return run_path, params_path, tensorboard_path


def make_submission_path(model_dir):
    # Get current date for saving submission
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    submission_dir = os.path.join('.', 'submissions', model_dir, date)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    submission_path = os.path.join(submission_dir, 'sub' + str(len(os.listdir(submission_dir))))
    return submission_path


def set_directories(model_dir, date, run):
    # Initialise all paths
    run_path = run_path = os.path.join('.', 'summaries', model_dir, date, 'run' + str(run))
    params_path = os.path.join(run_path, 'params')
    tensorboard_path = os.path.join(run_path, 'tensorboard')
    return run_path, params_path, tensorboard_path


def init_logger(log_dir):
    log_file = os.path.join(log_dir, 'train.log')
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter('%(message)s'))
    
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter('%(message)s'))
    
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger


def get_score(y_true, y_pred):
    auc_scores = []
    log_scores = []
    for i in range(y_true.shape[1]):
        auc_score = roc_auc_score(y_true[:, i], y_pred[:, i])
        auc_scores.append(auc_score)

        log_score = log_loss(y_true[:, i], y_pred[:, i])
        log_scores.append(log_score)
    avg_auc_score = np.mean(auc_scores)
    avg_loss_score = np.mean(log_scores)
    return avg_auc_score, auc_scores, avg_loss_score, log_scores

# def split_dataset_into_train_and_val(path_to_csv):
#     pass


# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated


# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes
