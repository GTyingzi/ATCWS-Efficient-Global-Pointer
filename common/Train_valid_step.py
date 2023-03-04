import torch
import config
from common.Metric import MetricsCalculator
from common.utils import attack_model

config = config.train_config
hyper_parameters = config["hyper_parameters"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metrics = MetricsCalculator()

def train_step(batch_train, model, optimizer, criterion):
    # batch_input_ids:(batch_size, seq_len)    batch_labels:(batch_size, ent_type_size, seq_len, seq_len)
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_train
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device)
                                                                                 )
    optimizer.zero_grad()
    logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

    loss = criterion(logits, batch_labels)
    loss.backward()

    if config["attack_train"] == "True": # 是否使用对抗训练
        attack_model(batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels,model,criterion,hyper_parameters["attack"])

    optimizer.step()

    return loss.item()


def valid_step(batch_valid, model):
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_valid
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device)
                                                                                 )
    with torch.no_grad():
        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
    sample_f1, sample_precision, sample_recall = metrics.get_evaluate_fpr(logits, batch_labels)

    return sample_f1, sample_precision, sample_recall

def test_step(batch_valid, model,metrics):
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_valid
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device)
                                                                                 )
    with torch.no_grad():
        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

    metrics.get_evaluate_temp2_fpr(logits,batch_labels)



###################引入分词信息###################################
def train_token_step(batch_train, model, optimizer, criterion,alpha=0.01):
    # batch_input_ids:(batch_size, seq_len)    batch_labels:(batch_size, ent_type_size, seq_len, seq_len)
    batch_samples, batch_seg_labels, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_train
    batch_seg_labels, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_seg_labels.to(device),
                                                                                 batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device)
                                                                                 )
    optimizer.zero_grad()
    overall_logits,logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids,token=True,istrain=True)
    loss = criterion(overall_logits, batch_labels) + alpha * criterion(logits,batch_seg_labels)
    loss.backward()

    if config["attack_train"] == "True": # 是否使用对抗训练
        attack_model(batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels,model,criterion,hyper_parameters["attack"])

    optimizer.step()

    return loss.item()

################### 测试可视化 ###################################################

def test_visual(batch_valid, model,metrics):
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_valid
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device)
                                                                                 )
    with torch.no_grad():
        overall_logits,logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        torch.save([batch_samples,logits,overall_logits],"visual/visual.pth")

    metrics.get_evaluate_temp2_fpr(logits,batch_labels)

def test_token_visual(batch_valid, model,metrics):
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_valid
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device)
                                                                                 )
    with torch.no_grad():
        overall_logits,logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids,token=True,istrain=True)
        torch.save([batch_samples,logits,overall_logits],"visual/visual_token.pth")

    metrics.get_evaluate_temp2_fpr(overall_logits, batch_labels)