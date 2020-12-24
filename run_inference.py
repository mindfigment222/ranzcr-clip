import torch
from utils import get_transforms, get_test_img_uids, make_submission_path
from datasets.test_dataset import TestDataset
from tqdm import tqdm
import numpy as np
from models.resnet50 import CustomResnext
import os


def run_inference(init_model, params_path, pretrained=False):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = init_model(num_classes=11, pretrained=pretrained)
    checkpoint = torch.load(params_path)
    model_dir = 'resnext50' # checkpoint['model_dir']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    root_dir = '/home/mszmelcz/Datasets/ranzcr-clip'
    transforms = get_transforms()['val']
    img_uids, submission_df = get_test_img_uids(os.path.join(root_dir, 'sample_submission.csv'))
    test_dataset = TestDataset(root=os.path.join(root_dir, 'test'), img_uids=img_uids, transforms=transforms) 

    # Parameters
    generator_params = {
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 6
    }

    test_generator = torch.utils.data.DataLoader(test_dataset, **generator_params)

    predictions = inference(model, test_generator, device)

    submission_df.iloc[:, 1:] = predictions
    submission_path = make_submission_path(model_dir)
    submission_df.to_csv(submission_path, index=False)
    print(submission_df.head())


def inference(model, test_generator, device):
    model.to(device)
    model.eval()

    predictions = []
    for inputs in tqdm(test_generator):
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=0).cpu().numpy()
        
        predictions.append(probs)
    predictions = np.concatenate(predictions)
    return predictions


if __name__ == '__main__':
    run_inference(CustomResnext, params_path='./summaries/resnext50/2020-12-23/run0/params/best.pth')





                         