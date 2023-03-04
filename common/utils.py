import torch
import numpy as np
import random
from common.attack_train import FGM,FGSM,FreeAT,PGD
import jieba

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def attack_model(batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels,model,criterion,attack_name):
    if attack_name == "fgm":
        fgm = FGM(model=model)
        fgm.attack()
        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        loss = criterion(logits, batch_labels)
        loss.backward()
        fgm.restore()

    elif attack_name == "fgsm":
        fgsm = FGSM(model=model)
        fgsm.attack()
        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        loss = criterion(logits, batch_labels)
        loss.backward()
        fgsm.restore()

    elif attack_name == "pgd":
        pgd = PGD(model=model)
        pgd_k = 3
        pgd.backup_grad()
        for _t in range(pgd_k):
            pgd.attack(is_first_attack=(_t == 0))

            if _t != pgd_k - 1:
                model.zero_grad()
            else:
                pgd.restore_grad()

            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            loss = criterion(logits, batch_labels)
            loss.backward()
        pgd.restore()

    elif attack_name == "FreeAT":
        free_at = FreeAT(model=model)
        m = 5
        free_at.backup_grad()
        for _t in range(m):
            free_at.attack(is_first_attack=(_t == 0))

            if _t != m - 1:
                model.zero_grad()
            else:
                free_at.restore_grad()

            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            loss = criterion(logits, batch_labels)
            loss.backward()
        free_at.restore()