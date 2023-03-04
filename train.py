import os
import config
import sys
import torch
from transformers import BertModel
from common.utils import seed_everything
from common.loss import multilabel_categorical_crossentropy
from common.Data_Process import load_ent2id,data_generator,data_generator
from models.GlobalPointer import EffiGlobalPointer,GlobalPointer,EffiGlobalPointer_token
from tqdm import tqdm
import glob
import wandb
from Test import evaluate
import time
from common.Train_valid_step import train_step,valid_step,train_token_step

config = config.train_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0

# 随机种子
seed_everything(hyper_parameters["seed"])

# 监控日志
if config["logger"] == "wandb" and config["run_type"] == "train":
    # init wandb
    wandb.init(project="GlobalPointer_" + config["exp_name"],
               config=hyper_parameters  # Initialize config
               )
    wandb.run.name = config["run_name"] + "_" + wandb.run.id

    model_state_dict_dir = wandb.run.dir
    logger = wandb
else:
    model_state_dict_dir = os.path.join(config["path_to_save_model"], config["exp_name"],
                                        time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime()))
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)


ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
ent2id = load_ent2id(ent2id_path)
ent_type_size = len(ent2id)

# 模型选择
encoder = BertModel.from_pretrained(config["bert_path"])
# model = GlobalPointer(encoder, ent_type_size, 64)
# model = EffiGlobalPointer(encoder, ent_type_size, 64)

model = EffiGlobalPointer_token(encoder, ent_type_size, 64)
model = model.to(device)

if config["logger"] == "wandb" and config["run_type"] == "train":
    wandb.watch(model)


def train(model, dataloader, epoch, optimizer):
    model.train()

    # loss func
    def loss_fun(y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        seq_len = y_pred.shape[-1]
        y_true = y_true.reshape(-1,seq_len * seq_len)
        y_pred = y_pred.reshape(-1,seq_len * seq_len)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss

    # scheduler
    if hyper_parameters["scheduler"] == "CAWR":
        T_mult = hyper_parameters["T_mult"]
        rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(dataloader) * rewarm_epoch_num,
                                                                         T_mult)
    elif hyper_parameters["scheduler"] == "Step":
        decay_rate = hyper_parameters["decay_rate"]
        decay_steps = hyper_parameters["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.
    for batch_ind, batch_data in pbar:

        # loss = train_step(batch_data, model, optimizer, loss_fun)
        loss = train_token_step(batch_data, model, optimizer, loss_fun)

        total_loss += loss

        avg_loss = total_loss / (batch_ind + 1)
        scheduler.step()

        pbar.set_description(f'Project:{config["exp_name"]}, Epoch: {epoch + 1}/{hyper_parameters["epochs"]}, Step: {batch_ind + 1}/{len(dataloader)}')
        pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])

        if config["logger"] == "wandb" and batch_ind % config["log_interval"] == 0:
            logger.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

def valid(model, dataloader):
    model.eval()

    total_f1, total_precision, total_recall = 0., 0., 0.
    for batch_data in tqdm(dataloader, desc="Validating"):
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
    if config["logger"] == "wandb":
        logger.log({"valid_precision": avg_precision, "valid_recall": avg_recall, "valid_f1": avg_f1})
    return avg_f1,avg_precision,avg_recall


if __name__ == '__main__':
    if config["run_type"] == "train":
        train_dataloader, val_dataloader = data_generator(data_type="train",exp_name=config["exp_name"]),\
                                           data_generator(data_type='val',exp_name=config["exp_name"])

        # optimizer
        init_learning_rate = float(hyper_parameters["lr"])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

        max_f1 = 0.
        for epoch in range(hyper_parameters["epochs"]):
            train(model, train_dataloader, epoch, optimizer)
            valid_f1,_,_ = valid(model, val_dataloader)
            if valid_f1 > max_f1:
                max_f1 = valid_f1
                if valid_f1 > config["f1_2_save"]:  # save the best model
                    model_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                    torch.save(model.state_dict(),
                               os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(model_state_num)))
            print(f"Best F1: {max_f1}")
            print("******************************************")
            if config["logger"] == "wandb":
                logger.log({"Best_F1": max_f1})
    elif config["run_type"] == "test":
        evaluate()
