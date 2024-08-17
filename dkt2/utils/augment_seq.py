import random
import numpy as np
import math



def atdkt_kt_seqs(
    s_seq,
    r_seq,
    t_seq,
):
    historyratios = []
    right, total = 0, 0
    for i in range(0, len(s_seq)):
        if r_seq[i] == 1:
            right += 1
        total += 1
        historyratios.append(right / total)
    
    return historyratios

def dimkt_kt_seqs(
    q_seq,
    s_seq,
    r_seq,
    t_seq,
    sds,
    qds,
):
    
    qd_seq, sd_seq = [], []
    for q, s, r in zip(q_seq, s_seq, r_seq):
        if r != -1:
            qd_seq.append(qds[q-1])
            sd_seq.append(sds[s-1])
        else:
            qd_seq.append(1)
            sd_seq.append(1)

    

    return qd_seq, sd_seq

def dkt_forget_kt_seqs(
    q_seq,
    s_seq,
    r_seq,
    t_seq,
):
    def log2(t):
        import math
        return round(math.log(t+1, 2))
    repeated_gap, sequence_gap, past_counts = [], [], []
    dlastskill, dcount = dict(), dict()
    pret = None
    max_rgap, max_sgap, max_pcount = 0, 0, 0
    for s, t in zip(s_seq, t_seq):
        s, t = int(s), int(t)
        if s not in dlastskill or s == 0:
            curRepeatedGap = 0
        else:
            curRepeatedGap = log2((t - dlastskill[s]) / 1000 / 60) + 1 # minutes
        dlastskill[s] = t
        max_rgap = max(max_rgap, curRepeatedGap)
        repeated_gap.append(curRepeatedGap)
        if pret == None or t == 0:
            curLastGap = 0
        else:
            curLastGap = log2((t - pret) / 1000 / 60) + 1
        pret = t
        max_sgap = max(max_sgap, curLastGap)
        sequence_gap.append(curLastGap)

        dcount.setdefault(s, 0)
        max_pcount = max(max_pcount, log2(dcount[s]))
        past_counts.append(log2(dcount[s]))
        dcount[s] += 1
    return repeated_gap, sequence_gap, past_counts, max_rgap, max_sgap, max_pcount


def replace_only(
    q_seq,
    s_seq,
    r_seq,
    replace_prob,
    easier_skills,
    harder_skills,
    q_mask_id,
    s_mask_id,
    seq_len,
    seed=None,
):
    rng = random.Random(seed)
    np.random.seed(seed)

    masked_q_seq = []
    masked_s_seq = []
    masked_r_seq = []
    replace_mask = []

    """
    skill difficulty based replace
    """
    if replace_prob > 0:
        for i, elem in enumerate(zip(s_seq, r_seq)):
            s, r = elem
            prob = rng.random()
            if prob < replace_prob and s != 0 and s != s_mask_id:
                replace_mask.append(r)
                if (
                    r == 0 and s in harder_skills
                ):  # if the response is wrong, then replace a skill with the harder one
                    masked_s_seq.append(harder_skills[s])
                elif (
                    r == 1 and s in easier_skills
                ):  # if the response is correct, then replace a skill with the easier one
                    masked_s_seq.append(easier_skills[s])
            else:
                masked_s_seq.append(s)
                replace_mask.append(-1)

    return masked_q_seq, masked_s_seq, masked_r_seq, replace_mask


def augment_kt_seqs(
    q_seq,
    s_seq,
    r_seq,
    mask_prob,
    crop_prob,
    permute_prob,
    replace_prob,
    negative_prob,
    easier_skills,
    harder_skills,
    q_mask_id,
    s_mask_id,
    seq_len,
    seed=None,
    skill_rel=None,
):
    rng = random.Random(seed)
    np.random.seed(seed)

    masked_q_seq = []
    masked_s_seq = []
    masked_r_seq = []
    negative_r_seq = []

    if mask_prob > 0:
        for q, s, r in zip(q_seq, s_seq, r_seq):
            prob = rng.random()
            if prob < mask_prob and s != 0:
                prob /= mask_prob
                if prob < 0.8:
                    masked_q_seq.append(q_mask_id)
                    masked_s_seq.append(s_mask_id)
                elif prob < 0.9:
                    masked_q_seq.append(
                        rng.randint(1, q_mask_id - 1)
                    ) 
                    masked_s_seq.append(
                        rng.randint(1, s_mask_id - 1)
                    ) 
                else:
                    masked_q_seq.append(q)
                    masked_s_seq.append(s)
            else:
                masked_q_seq.append(q)
                masked_s_seq.append(s)
            masked_r_seq.append(r)

            # reverse responses
            neg_prob = rng.random()
            if neg_prob < negative_prob and r != -1:  # padding
                negative_r_seq.append(1 - r)
            else:
                negative_r_seq.append(r)
    else:
        masked_q_seq = q_seq[:]
        masked_s_seq = s_seq[:]
        masked_r_seq = r_seq[:]

        for r in r_seq:
            # reverse responses
            neg_prob = rng.random()
            if neg_prob < negative_prob and r != -1:  # padding
                negative_r_seq.append(1 - r)
            else:
                negative_r_seq.append(r)

    """
    skill difficulty based replace
    """
    # print(harder_skills)
    if replace_prob > 0:
        for i, elem in enumerate(zip(masked_s_seq, masked_r_seq)):
            s, r = elem
            prob = rng.random()
            if prob < replace_prob and s != 0 and s != s_mask_id:
                if (
                    r == 0 and s in harder_skills
                ):  # if the response is wrong, then replace a skill with the harder one
                    masked_s_seq[i] = harder_skills[s]
                elif (
                    r == 1 and s in easier_skills
                ):  # if the response is correct, then replace a skill with the easier one
                    masked_s_seq[i] = easier_skills[s]

    true_seq_len = np.sum(np.asarray(q_seq) != 0)
    if permute_prob > 0:
        reorder_seq_len = math.floor(permute_prob * true_seq_len)
        start_idx = (np.asarray(q_seq) != 0).argmax()
        while True:
            start_pos = rng.randint(start_idx, seq_len - reorder_seq_len)
            if start_pos + reorder_seq_len < seq_len:
                break

        # reorder (permute)
        perm = np.random.permutation(reorder_seq_len)
        masked_q_seq = (
            masked_q_seq[:start_pos]
            + np.asarray(masked_q_seq[start_pos : start_pos + reorder_seq_len])[
                perm
            ].tolist()
            + masked_q_seq[start_pos + reorder_seq_len :]
        )
        masked_s_seq = (
            masked_s_seq[:start_pos]
            + np.asarray(masked_s_seq[start_pos : start_pos + reorder_seq_len])[
                perm
            ].tolist()
            + masked_s_seq[start_pos + reorder_seq_len :]
        )
        masked_r_seq = (
            masked_r_seq[:start_pos]
            + np.asarray(masked_r_seq[start_pos : start_pos + reorder_seq_len])[
                perm
            ].tolist()
            + masked_r_seq[start_pos + reorder_seq_len :]
        )

    # To-Do: check this crop logic!
    if 0 < crop_prob < 1:
        crop_seq_len = math.floor(crop_prob * true_seq_len)
        if crop_seq_len == 0:
            crop_seq_len = 1
        start_idx = (np.asarray(q_seq) != 0).argmax()
        while True:
            start_pos = rng.randint(start_idx, seq_len - crop_seq_len)
            if start_pos + crop_seq_len < seq_len:
                break

        masked_q_seq = masked_q_seq[start_pos : start_pos + crop_seq_len]
        masked_s_seq = masked_s_seq[start_pos : start_pos + crop_seq_len]
        masked_r_seq = masked_r_seq[start_pos : start_pos + crop_seq_len]

    pad_len = seq_len - len(masked_q_seq)

    attention_mask = [0] * pad_len + [1] * len(masked_s_seq)
    masked_q_seq = [0] * pad_len + masked_q_seq
    masked_s_seq = [0] * pad_len + masked_s_seq
    masked_r_seq = [-1] * pad_len + masked_r_seq

    return masked_q_seq, masked_s_seq, masked_r_seq, negative_r_seq, attention_mask


def preprocess_qr(questions, responses, seq_len, pad_val=-1):
    """
    split the interactions whose length is more than seq_len
    """
    preprocessed_questions = []
    preprocessed_responses = []

    for q, r in zip(questions, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i : i + seq_len])
            preprocessed_responses.append(r[i : i + seq_len])

            i += seq_len

        preprocessed_questions.append(
            np.concatenate([q[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_responses.append(
            np.concatenate([r[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )

    return preprocessed_questions, preprocessed_responses


def preprocess_qsr(questions, skills, responses, seq_len, pad_val=0):
    """
    split the interactions whose length is more than seq_len
    """
    preprocessed_questions = []
    preprocessed_skills = []
    preprocessed_responses = []
    attention_mask = []

    for q, s, r in zip(questions, skills, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i : i + seq_len])
            preprocessed_skills.append(s[i : i + seq_len])
            preprocessed_responses.append(r[i : i + seq_len])
            attention_mask.append(np.ones(seq_len))
            i += seq_len

        preprocessed_questions.append(
            np.concatenate([q[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_skills.append(
            np.concatenate([s[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_responses.append(
            np.concatenate([r[i:], np.array([-1] * (i + seq_len - len(q)))])
        )
        attention_mask.append(
            np.concatenate(
                [np.ones_like(r[i:]), np.array([0] * (i + seq_len - len(q)))]
            )
        )

    return (
        preprocessed_questions,
        preprocessed_skills,
        preprocessed_responses,
        attention_mask,
    )
