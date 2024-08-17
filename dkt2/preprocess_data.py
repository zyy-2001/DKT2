# This code is based on the following repositories:
#  1. https://github.com/theophilee/learner-performance-prediction/blob/master/prepare_data.py
#  2. https://github.com/THUwangcy/HawkesKT/blob/main/data/Preprocess.ipynb

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import os
import pickle
import time


# Please specify your dataset Path
BASE_PATH = "./dataset"
np.random.seed(12405)

def prepare_assistments(
    data_name: str, min_user_inter_num: int, remove_nan_skills: bool
):
    """
    Preprocess ASSISTments dataset

        :param data_name: (str) "assistments09", "assistments12", "assisments15", and "assistments17"
        :param min_user_inter_num: (int) Users whose number of interactions is less than min_user_inter_num will be removed
        :param remove_nan_skills: (bool) if True, remove interactions with no skill tage
        :param train_split: (float) proportion of data to use for training
        
        :output df: (pd.DataFrame) preprocssed ASSISTments dataset with user_id, item_id, timestamp, correct and unique skill features
        :output question_skill_rel: (csr_matrix) corresponding question-skill relationship matrix
    """
    data_path = os.path.join(BASE_PATH, data_name)
    df = pd.read_csv(os.path.join(data_path, "data.csv"), encoding="ISO-8859-1")

    # Only 2012 and 2017 versions have timestamps
    if data_name == "assistments09":
        # df = pd.read_csv(os.path.join(data_path, "skill_builder_data_corrected.csv"), encoding="ISO-8859-1")
        df = df.rename(columns={"problem_id": "item_id"})
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments12":
        # df = pd.read_csv(os.path.join(data_path, "2012-2013-data-with-predictions-4-final.csv"), encoding="ISO-8859-1")
        df = df.rename(columns={"problem_id": "item_id"})
        from datetime import datetime
        def change2timestamp(t, hasf=True):
            if hasf:
                timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
            else:
                timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
            return int(timeStamp)
        df["timestamp"] = df['start_time'].apply(lambda x:change2timestamp(x,hasf='.' in x))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()
        df["timestamp"] = (
            df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
        )
    elif data_name == "assistments15":
        df = df.rename(columns={"sequence_id": "item_id"})
        df["skill_id"] = df["item_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments17":
        df = df.rename(
            columns={
                "startTime": "timestamp",
                "studentId": "user_id",
                "problemId": "item_id",
                "skill": "skill_id",
            }
        )
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.loc[df["skill_id"].isnull(), "skill_id"] = -1

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_user_inter_num)

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"].astype(str), return_inverse=True)[1]
    # if data_name != 'assistments15' and data_name != 'assistments17' and data_name != 'assistments12':
    #     with open(os.path.join(data_path, "skill_id_name"), "wb") as f:
    #         pickle.dump(dict(zip(df["skill_id"], df["skill_name"])), f)

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Remove row duplicates due to multiple skills for one item
    if data_name == "assistments09":
        df = df.drop_duplicates("order_id")
    elif data_name == "assistments17":
        df = df.drop_duplicates(["user_id", "timestamp"])

    print("# Users: {}".format(df["user_id"].nunique()))
    print("# Skills: {}".format(df["skill_id"].nunique()))
    print("# Items: {}".format(df["item_id"].nunique()))
    print("# Interactions: {}".format(len(df)))

    # import sys
    # sys.exit()

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    print("# Preprocessed Skills: {}".format(df["skill_id"].nunique()))
    # Sort data temporally
    if data_name in ["assistments12", "assistments17"]:
        df.sort_values(by="timestamp", inplace=True)
    elif data_name == "assistments09":
        df.sort_values(by="order_id", inplace=True)
    elif data_name == "assistments15":
        df.sort_values(by="log_id", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])
    df.to_csv(os.path.join(data_path, "original_df.csv"), sep="\t", index=False)

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]

    df.reset_index(inplace=True, drop=True)

    user_num = df["user_id"].nunique()
    skill_num = df["skill_id"].nunique()
    u_skill_all_num = 0
    user_sparsity_5 = 0
    user_sparsity_10 = 0
    user_sparsity_20 = 0
    for _, udf in df.groupby("user_id"):
        u_skill_num = udf["skill_id"].nunique()
        u_skill_all_num += u_skill_num
        user_sparsity_5 += int(u_skill_num <= skill_num * 0.05)
        user_sparsity_10 += int(u_skill_num <= skill_num * 0.1)
        user_sparsity_20 += int(u_skill_num <= skill_num * 0.2)
        
    print(f'# Sparsity: {((user_num * skill_num - u_skill_all_num) / (user_num * skill_num) * 100):.2f}%')
    print(f'# User_sparsity_ratio_5: {(user_sparsity_5 / user_num * 100):.2f}%')
    print(f'# User_sparsity_ratio_10: {(user_sparsity_10 / user_num * 100):.2f}%')
    print(f'# User_sparsity_ratio_20: {(user_sparsity_20 / user_num * 100):.2f}%')

    # Save data
    with open(os.path.join(data_path, "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)



def prepare_patdisc(
    data_name: str, min_user_inter_num: int, remove_nan_skills: bool
):
    """
    Preprocess PATDisc dataset

        :param data_name: (str) "prob", "linux", "comp", and "database"
        :param min_user_inter_num: (int) Users whose number of interactions is less than min_user_inter_num will be removed
        :param remove_nan_skills: (bool) if True, remove interactions with no skill tage
        :param train_split: (float) proportion of data to use for training
        
        :output df: (pd.DataFrame) preprocssed PATDisc dataset with user_id, item_id, timestamp, correct and unique skill features
        :output question_skill_rel: (csr_matrix) corresponding question-skill relationship matrix
    """
    data_path = os.path.join(BASE_PATH, data_name)
    df = pd.read_csv(os.path.join(data_path, "processed_data.csv"), encoding="ISO-8859-1")

    
    df = df.rename(
        columns={
            "create_at": "timestamp",
            "user_id_new": "user_id",
            "problem_id_new": "item_id",
            "skill_id_new": "skill_id",
            "score": "correct"
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["timestamp"] = (
        df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    )
    
    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.loc[df["skill_id"].isnull(), "skill_id"] = -1

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_user_inter_num)

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"].astype(str), return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1


    print("# Users: {}".format(df["user_id"].nunique()))
    print("# Skills: {}".format(df["skill_id"].nunique()))
    print("# Items: {}".format(df["item_id"].nunique()))
    print("# Interactions: {}".format(len(df)))

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    print("# Preprocessed Skills: {}".format(df["skill_id"].nunique()))
    # Sort data temporally
    df.sort_values(by="timestamp", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])
    df.to_csv(os.path.join(data_path, "original_df.csv"), sep="\t", index=False)

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]

    df.reset_index(inplace=True, drop=True)

    # Save data
    with open(os.path.join(data_path, "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)

def prepare_ednet(min_user_inter_num, kc_col_name, remove_nan_skills):
    import re
    from tqdm import tqdm
    from IPython import embed 
    # timestamp,solving_id,question_id,user_answer,elapsed_time 
    # user_id	item_id	timestamp	correct	skill_id
    df_path = os.path.join(os.path.join(BASE_PATH, "ednet/KT1/"))
    user_path_list = os.listdir(df_path)
    print(f"total_user:{len(user_path_list)}") #784,309
    np.random.shuffle(user_path_list)

    content_path = os.path.join(os.path.join(BASE_PATH, "ednet/contents/questions.csv"))
    content_df = pd.read_csv(content_path)

    df = pd.DataFrame()

    count = 0

    for idx, user_path in enumerate(tqdm(user_path_list, total=len(user_path_list), ncols=50)):
        try:
            u_df = pd.read_csv(os.path.join(df_path, user_path), encoding = 'ISO-8859-1', dtype=str)
            if len(u_df) < min_user_inter_num : continue 
            
            uid = user_path.split('/')[-1]
            uid = int(re.sub(r'[^0-9]', '', uid))
            #get user_id
            u_df["user_id"] = uid

            
            all_questions = content_df["question_id"]
            user_questions = u_df["question_id"]
            u_df = u_df[user_questions.isin(all_questions)].dropna()
            
            #get skill_id
            skill_df = pd.merge(u_df, content_df.loc[:,["question_id", "correct_answer", "tags"]], how='outer', on="question_id").dropna()
            #get correct
           
            actual_ans = skill_df["correct_answer"].values
            user_ans = skill_df["user_answer"].values
            
            skill_df['correct'] = np.array(actual_ans == user_ans).astype(int)
            correct_count = skill_df['correct'].sum()
            total_count = len(skill_df)
            correct_rate = correct_count / total_count

            count += 1
            
            df = pd.concat([df, u_df])
            if count >= 20000 : break

        except:
            continue
    all_questions = content_df["question_id"]
    user_questions = df["question_id"]
    df = df[user_questions.isin(all_questions)].dropna()
    #get skill_id
    skill_df = pd.merge(df, content_df.loc[:,["question_id", "correct_answer", "tags"]], how='outer', on="question_id").dropna()
    #get correct
    actual_ans = skill_df["correct_answer"].values
    user_ans = skill_df["user_answer"].values
    skill_df['correct'] = np.array(actual_ans == user_ans).astype(int)

    #get item_id
    skill_df["item_id"] = skill_df["question_id"].str.extract(r'(\d+)')

    # Extract KCs
    kc_list = []
    for kc_str in skill_df["tags"].unique():
        for kc in kc_str.split(";"):
            kc_list.append(kc)
    kc_set = set(kc_list)
    kc2idx = {kc: i for i, kc in enumerate(kc_set)}

    # Adujust dtypes
    skill_df = skill_df.astype(
        {"correct": np.float64, "timestamp": np.float64}
    )

    # user, item, skill re-index
    skill_df["user_id"] = np.unique(skill_df["user_id"], return_inverse=True)[1]
    skill_df["item_id"] = np.unique(skill_df["item_id"], return_inverse=True)[1]
    skill_df["skill_id"] = np.unique(skill_df["tags"], return_inverse=True)[1]
    
    print("# Users: {}".format(skill_df["user_id"].nunique()))
    print("# Skills: {}".format(len(kc2idx)))
    print("# Preprocessed Skills: {}".format(skill_df["skill_id"].nunique()))
    print("# Items: {}".format(skill_df["item_id"].nunique()))
    print("# Interactions: {}".format(len(skill_df)))

    # Sort data temporally
    skill_df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)
    skill_df.sort_values(by="timestamp", inplace=True)

    print(f'skill_df: {skill_df.head()}')
    save_skill_df = skill_df[["timestamp", "user_id", "correct", "item_id", "skill_id"]].reset_index(drop=True)
    print(f'skill_df: {save_skill_df.head()}')
    data_path = os.path.join(BASE_PATH, "ednet/")
    save_skill_df.to_csv(os.path.join(data_path, "data.csv"), sep=",", index=False)
    
    

    

    

if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess DKT datasets")
    parser.add_argument("--data_name", type=str, default="assistments09")
    parser.add_argument("--min_user_inter_num", type=int, default=5)
    parser.add_argument("--remove_nan_skills", default=True, action="store_true")
    args = parser.parse_args()

    if args.data_name in [
        "assistments09",
        "assistments12",
        "assistments15",
        "assistments17",
    ]:
        prepare_assistments(
            data_name=args.data_name,
            min_user_inter_num=args.min_user_inter_num,
            remove_nan_skills=args.remove_nan_skills,
        )
    elif args.data_name in [
        "prob",
        "linux",
        "comp",
        "database"
    ]:
        prepare_patdisc(
            data_name=args.data_name,
            min_user_inter_num=args.min_user_inter_num,
            remove_nan_skills=args.remove_nan_skills
        )
    elif args.data_name == "ednet":
        prepare_ednet(
            min_user_inter_num=args.min_user_inter_num,
            kc_col_name="tags",
            remove_nan_skills=args.remove_nan_skills,
        )