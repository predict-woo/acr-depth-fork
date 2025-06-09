from typing import Any


from acr.main import ACR
from acr.config import args, parse_args, ConfigContext
import sys, os, cv2
from tqdm import tqdm
from acr.utils import save_results


def main():
    ################## Model Initialization ####################
    with ConfigContext(parse_args("configs/demo.yml")) as args_set:
        print("Loading the configurations from {}".format(args_set.configs_yml))
        acr = ACR(args_set=args_set)
        print(acr)

    ################## RUN on image forlder ####################
    results_dict = {}

    imgpath = "itw/book/0014.png"
    acr.output_dir = "final_out"

    os.makedirs(name=acr.output_dir, exist_ok=True)

    image = cv2.imread(imgpath)
    outputs = acr(image, imgpath)
    print(outputs)
    results_dict.update(outputs)

    if args().save_dict_results:
        save_results(imgpath, acr.output_dir, results_dict)
