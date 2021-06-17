from utils.system_utils import *
import sys
import warnings

from models import GeneralModel
from models.statistics.Metrics import Metrics
from utils.config_utils import *
from utils.model_utils import *
from utils.system_utils import *
import torch.nn.functional


def get_arguments():
    global arguments
    arguments = parse()
    if arguments.disable_autoconfig:
        autoconfig(arguments)
    return arguments


def load_checkpoint(model):
    path = os.path.join(RESULTS_DIR, "2021-04-14_16.12.24_example_runname", MODELS_DIR, "LeNet5_finished")
    state = DATA_MANAGER.load_python_obj(path)
    try:
        model.load_state_dict(state)
    except KeyError as e:
        print(list(state.keys()))
        raise e
    print("done")


""" validates the model on test set """

if __name__ == '__main__':
    device = "cuda"
    # get model
    model: GeneralModel = find_right_model(
        NETWORKS_DIR, "LeNet5",
        device="cuda",
        hidden_dim=512,
        input_dim=(1, 28, 28),
        output_dim=10,
        is_maskable=1,
        is_tracking_weights=0,
        is_rewindable=0,
        is_growable=0.0000 > 0,
        outer_layer_pruning=1,
        maintain_outer_mask_anyway=(
                                       not 1) and (
                                       0),
        l0=0,
        l0_reg=1.0,
        N=60000,
        beta_ema=0.999,
        l2_reg=5e-5
    ).to("cuda")
    arguments = get_arguments()
    load_checkpoint(model)
    print("\n")
    # load data
    train_loader, test_loader = find_right_model(
        DATASETS, "fashion",
        arguments=arguments
    )
    # init test mode
    import torch.nn.functional as F

    model.eval()
    cum_acc, cum_loss, cum_elapsed = [], [], []
    entropy = 0
    num = 0
    for batch_num, batch in enumerate(test_loader):
        x, y = batch
        x, y = x.to(device).float(), y.to(device)
        out = model(x)
        prob = F.softmax(out, dim=1)
        entropy -= torch.sum(prob * torch.log(prob))
        num += x.shape[0]
    entropy /= num
    print(entropy)
