import os
import config
import sys
import torch
from transformers import BertTokenizerFast, BertModel
from common.Data_Process import data_generator,load_ent2id
from models.GlobalPointer import GlobalPointer,EffiGlobalPointer,EffiGlobalPointer_token
from common.Metric import MetricsCalculator
from common.utils import seed_everything

from tqdm import tqdm
from common.Train_valid_step import valid_step,test_step,test_visual,test_token_visual

config = config.eval_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0

tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=True, do_lower_case=False)


ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
ent2id = load_ent2id(ent2id_path)
ent_type_size = len(ent2id)

metrics = MetricsCalculator()

def Test_1(model, dataloader):
    '''
    计算每个batch_size下的metric，在取每个batch_size的平均值
    '''
    model.eval()

    total_f1, total_precision, total_recall = 0., 0., 0.
    for batch_data in tqdm(dataloader, desc="Test"):
        f1, precision, recall = valid_step(batch_data, model)

        total_f1 += f1
        total_precision += precision
        total_recall += recall

    avg_f1 = total_f1 / (len(dataloader))
    avg_precision = total_precision / (len(dataloader))
    avg_recall = total_recall / (len(dataloader))
    print("******************************************")
    print(f'avg_precision: {avg_precision}, avg_recall: {avg_recall}, avg_f1: {avg_f1}')
    print("******************************************")

def Test_2(model, dataloader):
    '''
    计算全局的指标
    '''
    model.eval()
    for batch_data in tqdm(dataloader, desc="Test"):
        test_step(batch_data, model,metrics)

    f1, p, r = metrics.get_evaluate_2_fpr()
    print("******************************************")
    print(f'avg_precision: {p}, avg_recall: {r}, avg_f1: {f1}')
    print("******************************************")

def Test_3(model, dataloader):
    '''
    计算全局的指标，分别计算每个实体类比
    '''
    model.eval()
    for batch_data in tqdm(dataloader, desc="Test"):
        test_step(batch_data, model,metrics)
        # test_visual(batch_data, model,metrics) # 测试抽取、分类可视化
        # test_token_visual(batch_data, model,metrics)# 测试中文分词后的抽取、分类可视化
    label_metric,avg_metric = metrics.get_evaluate_2_label_fpr(ent2id)
    print("******************************************")
    print(label_metric)
    print("******************************************")
    print(avg_metric)

def load_model():
    model_state_dir = config["model_state_dir"]
    model_state_list = sorted(filter(lambda x: "model_state" in x, os.listdir(model_state_dir)),
                              key=lambda x: int(x.split(".")[0].split("_")[-1]))
    last_k_model = config["last_k_model"]
    model_state_path = os.path.join(model_state_dir, model_state_list[-last_k_model])
    encoder = BertModel.from_pretrained(config["bert_path"])

    # model = GlobalPointer(encoder, ent_type_size, 64)
    # model = EffiGlobalPointer(encoder, ent_type_size, 64)

    model = EffiGlobalPointer_token(encoder, ent_type_size, 64)

    model.load_state_dict(torch.load(model_state_path))
    model = model.to(device)

    return model

def evaluate():
    test_dataloader = data_generator(data_type="test",exp_name=config["exp_name"])
    model = load_model()

    # 计算f1指标
    Test_3(model,test_dataloader)