import argparse
import torch
from bunch import Bunch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from tester import Tester
from utils import losses
from utils.helpers import get_instance
from collections import OrderedDict


def main(data_path, weight_path, CFG, show):
    # Allow loading Bunch objects safely from checkpoint
    torch.serialization.add_safe_globals([Bunch])

    # Load checkpoint safely
    checkpoint = torch.load(weight_path, map_location="cpu")
    print("Checkpoint keys:", checkpoint.keys())  # Debugging info

    # Detect state_dict format
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # assume checkpoint is already a raw state_dict
        state_dict = checkpoint

    # Fix "module." prefix if checkpoint was saved with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    # Initialize model
    model = get_instance(models, 'model', CFG)
    model.load_state_dict(new_state_dict, strict=False)

    # Dataset and loader
    test_dataset = vessel_dataset(data_path, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Loss function
    loss = get_instance(losses, 'loss', checkpoint.get('config', {}))

    # Run Tester
    tester = Tester(model, loss, CFG, test_loader, data_path, show)
    tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default="/home/lwt/data_pro/vessel/DRIVE", type=str,
                        help="the path of dataset")
    parser.add_argument("-wp", "--weight_path", default="pretrained_weights/DRIVE/checkpoint-epoch40.pth", type=str,
                        help='the path of weight.pth')
    parser.add_argument("--show", help="save predict image",
                        required=False, default=False, action="store_true")
    args = parser.parse_args()

    # Load config.yaml safely and wrap in Bunch
    yaml = YAML(typ='safe', pure=True)
    with open('config.yaml', encoding='utf-8') as file:
        CFG = Bunch(yaml.load(file))

    main(args.dataset_path, args.weight_path, CFG, args.show)
