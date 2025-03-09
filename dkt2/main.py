import os
import argparse
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import yaml
from data_loaders import (
    MostRecentQuestionSkillDataset,
    MostEarlyQuestionSkillDataset,
    SimCLRDatasetWrapper,
    ATDKTDatasetWrapper,
    DimKTDatasetWrapper,
    DKTForgetDatasetWrapper,
)
from models.cl4kt import CL4KT
from models.dkt import DKT
from models.sakt import SAKT
from models.dkvmn import DKVMN
from models.simplekt import simpleKT
from models.akt import AKT
from models.dtransformer import DTransformer
from models.stablekt import stableKT
from models.atkt import ATKT
from models.folibikt import folibiKT
from models.skvmn import SKVMN
from models.deep_irt import DeepIRT
from models.sparsekt import sparseKT
from models.gkt import GKT
from models.gkt_utils import get_gkt_graph
from models.atdkt import ATDKT
from models.dimkt import DIMKT
from models.dkt_forget import DKTForget
from models.dkt_plus import DKTPlus
from models.bkt import BKT
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from utils.config import ConfigNode as CN
from utils.file_io import PathManager
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
if conda_env == 'mamba4kt':
    from models.mamba4kt import Mamba4KT
elif conda_env == 'dkt2':
    from models.dkt2 import DKT2

torch.backends.cudnn.enable=True
torch.backends.cudnn.benchmark=True

def main(config):
    accelerator = Accelerator()
    device = accelerator.device

    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name
    seed = config.seed
    trans = config.trans
    length = config.len
    mask_future = config.mask_future
    pred_last = config.pred_last
    mask_response = config.mask_response
    config_seq_len = config.seq_len
    joint = config.joint

    np.random.seed(seed)
    torch.manual_seed(seed)

    
    train_config = config.train_config
    checkpoint_dir = config.checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, data_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = train_config.batch_size
    eval_batch_size = train_config.eval_batch_size
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer
    seq_len = train_config.seq_len
    if config_seq_len != -1:
        seq_len = config_seq_len

    if train_config.sequence_option == "recent":  # the most recent N interactions
        dataset = MostRecentQuestionSkillDataset
    elif train_config.sequence_option == "early":  # the most early N interactions
        dataset = MostEarlyQuestionSkillDataset
    else:
        raise NotImplementedError("sequence option is not valid")

    test_aucs, test_accs, test_rmses = [], [], []

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    df_path = os.path.join(os.path.join(dataset_path, data_name), "preprocessed_df.csv")
    df = pd.read_csv(df_path, sep="\t")

    if model_name == 'dimkt':
        questions_difficult_path = os.path.join(os.path.join(dataset_path, data_name), "questions_difficult_100.csv")
        skills_difficult_path = os.path.join(os.path.join(dataset_path, data_name), "skills_difficult_100.csv")
        def difficult_compute(df, difficult_path, tag, diff_level=100):
            sd = {}
            df = df.reset_index(drop=True)
            set_tags = set(np.array(df[tag]))
            from tqdm import tqdm
            for i in tqdm(set_tags):
                count = 0
                idx = df[(df[tag] == i)].index.tolist()
                tmp_data = df.iloc[idx]
                correct_1 = tmp_data['correct']
                if len(idx) < 30:
                    sd[i] = 1
                    continue
                else:
                    for j in np.array(correct_1):
                        count += j
                    if count == 0:
                        sd[i] = 1
                        continue
                    else:
                        avg = int((count/len(correct_1))*diff_level)+1
                        sd[i] = avg
            with open(difficult_path,'w',newline='',encoding='UTF8') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(sd.keys())
                writer.writerow(sd.values())
            return

        if not os.path.exists(questions_difficult_path):
            difficult_compute(df, questions_difficult_path, 'item_id')
        if not os.path.exists(skills_difficult_path):
            difficult_compute(df, skills_difficult_path, 'skill_id')

        sds = {}
        qds = {}
        import csv
        with open(skills_difficult_path,'r',encoding="UTF8") as f:
            reader = csv.reader(f)
            sds_keys = next(reader)
            sds_vals = next(reader)
            for i in range(len(sds_keys)):
                sds[int(sds_keys[i])] = int(sds_vals[i])
        with open(questions_difficult_path,'r',encoding="UTF8") as f:
            reader = csv.reader(f)
            qds_keys = next(reader)
            qds_vals = next(reader)
            for i in range(len(qds_keys)):
                qds[int(qds_keys[i])] = int(qds_vals[i])

    print("skill_min", df["skill_id"].min())
    users = df["user_id"].unique()
    df["skill_id"] += 1  # zero for padding
    df["item_id"] += 1  # zero for padding
    num_skills = df["skill_id"].max() + 1
    num_questions = df["item_id"].max() + 1

    np.random.shuffle(users)

    print("MODEL", model_name)
    print(dataset)
    if data_name in ["statics", "assistments15"]:
        num_questions = 0

    for fold, (train_ids, test_ids) in enumerate(kfold.split(users)):
        if model_name == "cl4kt":
            model_config = config.cl4kt_config
            model = CL4KT(joint, mask_response, pred_last, mask_future, length, trans, num_skills, num_questions, seq_len, **model_config)
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob
        elif model_name == 'dkt':
            model_config = config.dkt_config
            model = DKT(joint, mask_future, length, num_skills, **model_config)
        elif model_name == 'sakt':
            model_config = config.sakt_config
            model = SAKT(joint, mask_future, length, trans, num_skills, seq_len, **model_config)
        elif model_name == 'dkvmn':
            model_config = config.dkvmn_config
            model = DKVMN(joint, mask_response, pred_last, mask_future, length, trans, num_skills, **model_config)
        elif model_name == 'skvmn':
            model_config = config.skvmn_config
            model = SKVMN(pred_last, mask_future, length, trans, num_skills, **model_config)
        elif model_name == 'deep_irt':
            model_config = config.deep_irt_config
            model = DeepIRT(mask_response, pred_last, mask_future, length, trans, num_skills, **model_config)
        elif model_name == 'simplekt':
            model_config = config.simplekt_config
            model = simpleKT(mask_response, pred_last, mask_future, length, trans, num_skills, num_questions, seq_len, **model_config)
        elif model_name == "akt":
            model_config = config.akt_config
            if data_name in ["statics", "assistments15"]:
                num_questions = 0
            model = AKT(joint, mask_response, pred_last, mask_future, length, trans, num_skills, num_questions, **model_config)
        elif model_name == 'atkt':
            model_config = config.atkt_config
            model = ATKT(joint, mask_future, length, num_skills, **model_config)
        elif model_name == 'folibikt':
            model_config = config.folibikt_config
            model = folibiKT(mask_response, pred_last, mask_future, length, trans, num_skills, num_questions, seq_len, **model_config)
        elif model_name == "sparsekt":
            model_config = config.sparsekt_config
            model = sparseKT(mask_response, pred_last, mask_future, length, trans, num_skills, num_questions, seq_len, **model_config)
        elif model_name == 'gkt':
            model_config = config.gkt_config
            graph_type = model_config['graph_type']
            fname = f"gkt_graph_{graph_type}.npz"
            graph_path = os.path.join(os.path.join(dataset_path, data_name), fname)
            if os.path.exists(graph_path):
                graph = torch.tensor(np.load(graph_path, allow_pickle=True)['matrix']).float()
            else:
                graph = get_gkt_graph(df, num_skills, graph_path, graph_type=graph_type)
                graph = torch.tensor(graph).float()
            model = GKT(length, device, num_skills, graph, **model_config)
        elif model_name == 'dtransformer':
            model_config = config.dtransformer_config
            if data_name in ["statics", "assistments15"]:
                num_questions = 0
            model = DTransformer(mask_response, pred_last, mask_future, length, trans, num_skills, num_questions, **model_config)
        elif model_name == 'stablekt':
            if data_name in ["statics", "assistments15"]:
                num_questions = 0
            model_config = config.stablekt_config
            model = stableKT(mask_response, pred_last, mask_future, length, trans, num_skills, num_questions, **model_config)
        elif model_name == 'atdkt':
            model_config = config.atdkt_config
            model = ATDKT(joint, mask_future, length, num_skills, num_questions, **model_config)
        elif model_name == 'dimkt':
            model_config = config.dimkt_config
            model = DIMKT(mask_future, length, trans, num_skills, num_questions, **model_config)
        elif model_name == 'dkt2':
            model_config = config.dkt2_config
            model = DKT2(joint, mask_future, length, num_skills, num_questions, batch_size, seq_len, device, **model_config)
        elif model_name == 'dkt_plus':
            model_config = config.dkt_plus_config
            model = DKTPlus(mask_future, length, num_skills, **model_config)
        elif model_name == 'bkt':
            model_config = config.bkt_config
            model = BKT(length, **model_config)
        elif model_name == 'mamba4kt':
            model_config = config.mamba4kt_config
            model = Mamba4KT(joint, length, num_skills, num_questions, **model_config)

        train_users = users[train_ids]
        np.random.shuffle(train_users)
        offset = int(len(train_ids) * 0.9)

        valid_users = train_users[offset:]
        train_users = train_users[:offset]

        test_users = users[test_ids]

        train_df = df[df["user_id"].isin(train_users)]
        valid_df = df[df["user_id"].isin(valid_users)]
        test_df = df[df["user_id"].isin(test_users)]

        
        train_dataset = dataset(train_df, seq_len, num_skills, num_questions)
        valid_dataset = dataset(valid_df, seq_len, num_skills, num_questions)
        test_dataset = dataset(test_df, seq_len, num_skills, num_questions)

        print("train_ids", len(train_users))
        print("valid_ids", len(valid_users))
        print("test_ids", len(test_users))

        if model_name == 'dkt_forget':
            num_rgap, num_sgap, num_pcount = 0, 0, 0
        if "cl" in model_name:  # contrastive learning
            train_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        train_dataset,
                        seq_len,
                        mask_prob,
                        crop_prob,
                        permute_prob,
                        replace_prob,
                        negative_prob,
                        eval_mode=False,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        valid_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        test_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )
        elif "atdkt" in model_name: # atdkt
            train_loader = accelerator.prepare(
                DataLoader(
                    ATDKTDatasetWrapper(
                        train_dataset,
                        seq_len,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    ATDKTDatasetWrapper(
                        valid_dataset,
                        seq_len,
                    ),
                    batch_size=eval_batch_size,
                )
            )
            
            test_loader = accelerator.prepare(
                DataLoader(
                    ATDKTDatasetWrapper(
                        test_dataset,
                        seq_len,
                    ),
                    batch_size=eval_batch_size,
                )
            )
        elif "dimkt" in model_name: # dimkt
            train_loader = accelerator.prepare(
                DataLoader(
                    DimKTDatasetWrapper(
                        train_dataset,
                        seq_len,
                        sds,
                        qds,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    DimKTDatasetWrapper(
                        valid_dataset,
                        seq_len,
                        sds,
                        qds,
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    DimKTDatasetWrapper(
                        test_dataset,
                        seq_len,
                        sds,
                        qds,
                    ),
                    batch_size=eval_batch_size,
                )
            )
        elif "dkt_forget" in model_name: # dkt_forget
            dkt_forget_train_dataset = DKTForgetDatasetWrapper(
                        train_dataset,
                        seq_len,
                    )
            train_loader = accelerator.prepare(
                DataLoader(
                    dkt_forget_train_dataset,
                    batch_size=batch_size,
                )
            )
            dkt_forget_valid_dataset = DKTForgetDatasetWrapper(
                        valid_dataset,
                        seq_len,
                    )
            valid_loader = accelerator.prepare(
                DataLoader(
                    dkt_forget_valid_dataset,
                    batch_size=eval_batch_size,
                )
            )
            dkt_forget_test_dataset = DKTForgetDatasetWrapper(
                        test_dataset,
                        seq_len,
                    )
            test_loader = accelerator.prepare(
                DataLoader(
                    dkt_forget_test_dataset,
                    batch_size=eval_batch_size,
                )
            )
            num_rgap = max(num_rgap, dkt_forget_train_dataset.max_rgap)
            num_sgap = max(num_sgap, dkt_forget_train_dataset.max_sgap)
            num_pcount = max(num_pcount, dkt_forget_train_dataset.max_pcount)
            num_rgap = max(num_rgap, dkt_forget_valid_dataset.max_rgap)
            num_sgap = max(num_sgap, dkt_forget_valid_dataset.max_sgap)
            num_pcount = max(num_pcount, dkt_forget_valid_dataset.max_pcount)
            num_rgap = max(num_rgap, dkt_forget_test_dataset.max_rgap)
            num_sgap = max(num_sgap, dkt_forget_test_dataset.max_sgap)
            num_pcount = max(num_pcount, dkt_forget_test_dataset.max_pcount)

        else:
            train_loader = accelerator.prepare(
                DataLoader(train_dataset, batch_size=batch_size)
            )

            valid_loader = accelerator.prepare(
                DataLoader(valid_dataset, batch_size=eval_batch_size)
            )

            test_loader = accelerator.prepare(
                DataLoader(test_dataset, batch_size=eval_batch_size)
            )
        
        if model_name == 'dkt_forget':
            model_config = config.dkt_forget_config
            model = DKTForget(mask_future, length, device, num_skills, num_rgap+1, num_sgap+1, num_pcount+1, **model_config)

        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)
        if model_name != 'bkt':
            if optimizer == "sgd":
                opt = SGD(model.parameters(), learning_rate, momentum=0.9)
            elif optimizer == "adam":
                opt = Adam(model.parameters(), learning_rate, weight_decay=train_config.wl)

            model, opt = accelerator.prepare(model, opt)
        else:
            opt = None


        test_auc, test_acc, test_rmse = model_train(
            fold,
            model,
            accelerator,
            opt,
            train_loader,
            valid_loader,
            test_loader,
            config,
            n_gpu,
        )


        test_aucs.append(test_auc)
        test_accs.append(test_acc)
        test_rmses.append(test_rmse)

    test_auc = np.mean(test_aucs)
    test_auc_std = np.std(test_aucs)
    test_acc = np.mean(test_accs)
    test_acc_std = np.std(test_accs)
    test_rmse = np.mean(test_rmses)
    test_rmse_std = np.std(test_rmses)

    now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time
    if pred_last == True:
        log_out_path = os.path.join(
            os.path.join("logs", "5-fold-cv", "{}".format(data_name), "{}".format(trans), "{}".format(mask_future), "pred_last", "{}".format(length),)
        )
    elif mask_response == True:
        log_out_path = os.path.join(
            os.path.join("logs", "5-fold-cv", "{}".format(data_name), "{}".format(trans), "{}".format(mask_future), "mask_response", "{}".format(length),)
        )
    elif joint == True:
        log_out_path = os.path.join(
            os.path.join("logs", "5-fold-cv", "{}".format(data_name), "multi-concept", "{}".format(trans),)
        )
    elif config_seq_len != -1:
        log_out_path = os.path.join(
            os.path.join("logs", "5-fold-cv", "{}".format(data_name), "his_len_{}".format(config_seq_len), "{}".format(trans), "{}".format(length),)
        )
    else:
        log_out_path = os.path.join(
            os.path.join("logs", "5-fold-cv", "{}".format(data_name), "{}".format(trans), "{}".format(mask_future), "{}".format(length),)
        )
    os.makedirs(log_out_path, exist_ok=True)
    with open(os.path.join(log_out_path, "{}-{}".format(model_name, now)), "w") as f:
        f.write("AUC\tACC\tRMSE\tAUC_std\tACC_std\tRMSE_std\n")
        f.write("{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(test_auc, test_acc, test_rmse, test_auc_std, test_acc_std, test_rmse_std))
        f.write("AUC_ALL\n")
        f.write(",".join([str(auc) for auc in test_aucs])+"\n")
        f.write("ACC_ALL\n")
        f.write(",".join([str(auc) for auc in test_accs])+"\n")
        f.write("RMSE_ALL\n")
        f.write(",".join([str(auc) for auc in test_rmses])+"\n")

    print("\n5-fold CV Result")
    print("AUC\tACC\tRMSE")
    print("{:.5f}\t{:.5f}\t{:.5f}".format(test_auc, test_acc, test_rmse))


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt2",
        help="The name of the model to train. \
            The possible models are in [akt, cl4kt, dkt, dkt_forget, dkt_plus, sakt, simplekt, dkvmn...]. \
            The default model is dkt2.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="spanish",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--reg_cl",
        type=float,
        default=0.1,
        help="regularization parameter contrastive learning loss",
    )
    parser.add_argument("--mask_prob", type=float, default=0.2, help="mask probability")
    parser.add_argument("--crop_prob", type=float, default=0.3, help="crop probability")
    parser.add_argument(
        "--permute_prob", type=float, default=0.3, help="permute probability"
    )
    parser.add_argument(
        "--replace_prob", type=float, default=0.3, help="replace probability"
    )
    parser.add_argument(
        "--negative_prob",
        type=float,
        default=1.0,
        help="reverse responses probability for hard negative pairs",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout probability"
    )
    parser.add_argument(
        "--batch_size", type=float, default=512, help="train batch size"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=64, help="embedding size"
    )
    parser.add_argument(
        "--state_d", type=int, default=64, help="hidden size"
    )
    parser.add_argument(
        "--trans", type=str2bool, default=False, help="Convert the original incomplete output to a complete output setting"
    )
    parser.add_argument(
        "--len", type=int, default=1, help="Length of the predicted sequence"
    )
    parser.add_argument(
        "--mask_future", type=str2bool, default=False, help="Whether to mask out the future sequence in models like AKT and make predictions"
    )
    parser.add_argument(
        "--pred_last", type=str2bool, default=False, help="Only predict the last len questions"
    )
    parser.add_argument(
        "--mask_response", type=str2bool, default=False, help="Whether to mask out the future response sequence in models like AKT and make predictions"
    )
    parser.add_argument(
        "--seq_len", type=int, default=-1, help="Length of the history sequence"
    )
    parser.add_argument(
        "--joint", type=str2bool, default=False, help="Whether to predict multiple concepts at the same time"
    )
    parser.add_argument("--l2", type=float, default=1e-5, help="l2 regularization param")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    
    args = parser.parse_args()

    base_cfg_file = PathManager.open("configs/example.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.eval_batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer
    cfg.trans = args.trans
    cfg.len = args.len
    cfg.mask_future = args.mask_future
    cfg.pred_last = args.pred_last
    cfg.mask_response = args.mask_response
    cfg.seq_len = args.seq_len
    cfg.joint = args.joint
    print(f'trans: {args.trans}')
    print(f'prediction length: {args.len}')
    print(f'mask_future: {args.mask_future}')
    print(f'pred_last: {args.pred_last}')
    print(f'mask_response: {args.mask_response}')
    print(f'seq_len: {args.seq_len}')
    print(f'joint: {args.joint}')
    if cfg.trans == True:
        assert args.model_name not in ["dkt", "dkt_forget", "dkt_plus", "dkt2", "gkt", "atkt", "atdkt", "mamba4kt"]
        assert args.mask_future == False, "When configuring the inputs and outputs of models like AKT, the future sequence cannot be masked"
    if cfg.pred_last == True:
        assert args.model_name not in ["dkt", "dkt_forget", "dkt_plus", "dkt2", "gkt", "atkt", "atdkt", "mamba4kt"]
        assert args.trans == False and args.mask_future == False, "The input setting of the standard KT model only predicts the last len questions"
    if cfg.mask_response == True:
        assert args.model_name not in ["dkt", "dkt_forget", "dkt_plus", "dkt2", "gkt", "atkt", "atdkt", "mamba4kt"]
        assert args.trans == False and args.mask_future == False and args.pred_last == False
    if cfg.joint == True:
        if args.model_name not in ["dkt", "dkt_forget", "dkt_plus", "dkt2", "gkt", "atkt", "atdkt", "mamba4kt"]:
            assert args.trans == True, "All other models require a switch output setting when predicting multiple concepts"

    if args.model_name == "cl4kt":
        cfg.cl4kt_config.reg_cl = args.reg_cl
        cfg.cl4kt_config.mask_prob = args.mask_prob
        cfg.cl4kt_config.crop_prob = args.crop_prob
        cfg.cl4kt_config.permute_prob = args.permute_prob
        cfg.cl4kt_config.replace_prob = args.replace_prob
        cfg.cl4kt_config.negative_prob = args.negative_prob
        cfg.cl4kt_config.dropout = args.dropout
        cfg.cl4kt_config.l2 = args.l2
    elif args.model_name == 'dkt':  # dkt
        cfg.dkt_config.dropout = args.dropout
    elif args.model_name == 'sakt':  # sakt 
        cfg.sakt_config.dropout = args.dropout
    elif args.model_name == 'dkvmn':  # dkvmn
        cfg.dkvmn_config.dropout = args.dropout
    elif args.model_name == 'skvmn':  # skvmn
        cfg.skvmn_config.dropout = args.dropout
    elif args.model_name == 'deep_irt':  # deep_irt
        cfg.deep_irt_config.dropout = args.dropout
    elif args.model_name == 'simplekt':  # simplekt
        cfg.simplekt_config.dropout = args.dropout
    elif args.model_name == 'akt':  # akt
        cfg.akt_config.l2 = args.l2
        cfg.akt_config.dropout = args.dropout
    elif args.model_name == 'atkt':  # atkt
        cfg.atkt_config.dropout = args.dropout
    elif args.model_name == 'folibikt':  # folibikt
        cfg.folibikt_config.l2 = args.l2
        cfg.folibikt_config.dropout = args.dropout
    elif args.model_name == 'sparsekt':  # sparsekt
        cfg.sparsekt_config.dropout = args.dropout
    elif args.model_name == 'gkt':  # gkt
        cfg.gkt_config.dropout = args.dropout
    elif args.model_name == 'dtransformer':  # dtransformer
        cfg.dtransformer_config.dropout = args.dropout
        cfg.dtransformer_config.embedding_size = args.embedding_size
    elif args.model_name == 'stablekt':  # stablekt
        cfg.stablekt_config.dropout = args.dropout
        cfg.stablekt_config.embedding_size = args.embedding_size
    elif args.model_name == 'atdkt':  # atdkt
        cfg.atdkt_config.dropout = args.dropout
    elif args.model_name == 'dimkt':  # dimkt 
        cfg.dimkt_config.dropout = args.dropout
    elif args.model_name == 'dkt2':  # dkt2
        cfg.dkt2_config.dropout = args.dropout
    elif args.model_name == 'dkt_forget':  # dkt_forget
        cfg.dkt_forget_config.dropout = args.dropout
    elif args.model_name == 'dkt_plus':  # dkt_plus
        cfg.dkt_plus_config.dropout = args.dropout
    elif args.model_name == 'bkt':  # bkt
        pass
    elif args.model_name == 'mamba4kt':  # mamba4kt
        cfg.mamba4kt_config.dropout = args.dropout


    cfg.freeze()

    print(cfg)
    import time
    start_time = time.perf_counter()
    main(cfg)
    end_time = time.perf_counter()
    print(f'exec time: {round((end_time - start_time)*1000, 2)}ms')
    
