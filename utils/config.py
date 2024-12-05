import os
import time
import torch
import argparse

from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile, dict_to_markdown
import shutil

class BaseOptions(object):
    saved_option_filename = "opt.json"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        parser.add_argument("--dset_name", type=str, choices=["ADNI", 'ADNI_90_120_fMRI', 'FTD_90_200_fMRI', 'OCD_90_200_fMRI', 'PPMI'])
        parser.add_argument("--results_root", type=str, default="results")
        parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
        parser.add_argument("--seed", type=int, default=2018, help="random seed")
        parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        parser.add_argument("--no_pin_memory", action="store_true",
                            help="Don't use pin_memory=True for dataloader. "
                                 "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")
        parser.add_argument("--num_workers", type=int, default=4,
                            help="num subprocesses used to load the data, 0: use main process")
        # Data config
        parser.add_argument("--data_root", type=str, default="data")
        parser.add_argument("--label_id", type=dict, default=None, help="label to id mapping, We suggest to use default")
        parser.add_argument("--discrete", action="store_true", help="PLEASE DO NOT USE --discrete !!!")
        parser.add_argument("--slicenum", type=int, default=-1, help="PLEASE DO NOT USE --slicenum !!!")
        parser.add_argument("--truncate", type=int, default=4, help="truncate float to this decimal place")
        parser.add_argument("--ratio", type=float, default=0.2, help="mask ratio")
        # training config
        parser.add_argument("--model_name_or_path", type=str, default="./Llama-2-7b-hf",)
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--n_epoch", type=int, default=500, help="number of epochs to run")
        parser.add_argument("--bsz", type=int, default=16, help="mini-batch size")
        parser.add_argument("--eval_bsz", type=int, default=8,
                            help="mini-batch size at inference, for query")
        parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_all", action="store_true",
                            help="if --resume_all, load optimizer/scheduler/epoch as well")
        self.parser = parser

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print(dict_to_markdown(vars(opt), max_str_len=120))
        # Save settings
        option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
        save_json(args, option_file_path, save_pretty=True)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        opt.results_dir = os.path.join(opt.results_root,
                                        "-".join([opt.dset_name, opt.exp_id,
                                                    time.strftime("%Y_%m_%d_%H_%M_%S")]))
        mkdirp(opt.results_dir)

        self.display_save(opt)

        opt.device = torch.device("cuda" if opt.device >= 0 else "cpu")
        opt.pin_memory = not opt.no_pin_memory

        ########Init the label_id########
        if opt.label_id is None:
            opt.label_id = {}
            opt.label_id["ADNI"] = {
                "NC": 0,
                "AD": 1,
                "MCI": 2,
                "MCIn": 3,
                "MCIp": 4
            }
            opt.label_id["ADNI_90_120_fMRI"] = {
                "NC": 0,
                "AD": 1,
                "EMCI": 2,
                "LMCI": 3,
            }
            opt.label_id["FTD_90_200_fMRI"] = {
                "NC": 0,
                "FTD": 1,
            }
            opt.label_id["OCD_90_200_fMRI"] = {
                "NC": 0,
                "OCD": 1,
            }
            opt.label_id["PPMI"] = {
                "NC": 0,
                "PD": 1,
            }
        ########Init the label_id done########

        self.opt = opt
        return opt