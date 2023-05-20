import gc # garbage collection 
from turtle import forward
import numpy as np
import pandas as pd
import wandb 
wandb.login(key="670a0e7850bd4044cacba50ca610b7b66a3de782")
wandb.init(project='protac_project')

# ModuleNotFoundError 발생해서 다시 지정
# from utils.utils import *
# from utils.attention_flow import *
# from utils.utils import deleteEncodingLayers

from utils import *
from util import *
# from util import attention_flow
# from util import load_hparams

# import sys
# sys.path.append('/home/jina/PROTAC_NLP/PROTAC-Prediction')



import torch
import torch.nn as nn

import sklearn as sk
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoConfig, AutoTokenizer, RobertaModel, BertModel

from torchmetrics.functional import f1_score, accuracy, auroc, precision_recall_curve
from sklearn.metrics import f1_score, roc_curve, precision_score, recall_score, auc
from sklearn.metrics import roc_auc_score, average_precision_score


class ProtacDataset(Dataset):
    def __init__(self, list_IDs, labels, df_protac, d_tokenizer, poi_tokenizer, e3_tokenizer, prot_maxLength):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_protac

        self.d_tokenizer = d_tokenizer
        self.poi_tokenizer = poi_tokenizer 
        self.e3_tokenizer = e3_tokenizer
        
        self.prot_maxLength = prot_maxLength


    def convert_data(self, drug_data, poi_data, e3_data):
        poi_data = ' '.join(list(poi_data))
        e3_data = ' '.join(list(e3_data))
        
        d_inputs = self.d_tokenizer(drug_data, return_tensors="pt")
        poi_inputs = self.poi_tokenizer(poi_data, return_tensors="pt")
        e3_inputs = self.e3_tokenizer(e3_data, return_tensors="pt")

        # protac inputs
        drug_input_ids = d_inputs['input_ids']
        drug_attention_mask = d_inputs['attention_mask']
        drug_inputs = {'input_ids': drug_input_ids, 'attention_mask': drug_attention_mask}

        # poi inputs
        poi_input_ids = poi_inputs['input_ids']
        poi_attention_mask = poi_inputs['attention_mask']
        poi_inputs = {'input_ids': poi_input_ids, 'attention_mask': poi_attention_mask}
        
        # e3 inputs
        e3_input_ids = e3_inputs['input_ids']
        e3_attention_mask = e3_inputs['attention_mask']
        e3_inputs = {'input_ids': e3_input_ids, 'attention_mask': e3_attention_mask}
        
        return drug_inputs, poi_inputs, e3_inputs


    def tokenize_data(self, drug_data, poi_data, e3_data):
        poi_data = ' '.join(list(poi_data))
        e3_data = ' '.join(list(e3_data))

        tokenize_drug = ['[CLS]'] + self.d_tokenizer.tokenize(drug_data) + ['[SEP]']
        tokenize_poi = ['[CLS]'] + self.poi_tokenizer.tokenize(poi_data) + ['[SEP]']
        tokenize_e3 = ['[CLS]'] + self.e3_tokenizer.tokenize(e3_data) + ['[SEP]']

        return tokenize_drug, tokenize_poi, tokenize_e3


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)


    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        drug_data = self.df.iloc[index]['smiles']
        poi_data = self.df.iloc[index]['poi_residue']
        poi_data = ' '.join(list(poi_data))
        e3_data = self.df.iloc[index]['e3_residue']
        e3_data = ' '.join(list(e3_data))
        
        # TODO! d_inputs의 max_length 나중에 다시 설정해주기!!!
        d_inputs = self.d_tokenizer(drug_data, padding='max_length', max_length=510, truncation=True, return_tensors="pt")
        poi_inputs = self.poi_tokenizer(poi_data, padding='max_length', max_length=self.prot_maxLength, truncation=True, return_tensors="pt")
        e3_inputs = self.e3_tokenizer(e3_data, padding='max_length', max_length=self.prot_maxLength, truncation=True, return_tensors="pt")
        
        
        d_input_ids = d_inputs['input_ids'].squeeze()
        d_attention_mask = d_inputs['attention_mask'].squeeze()
        poi_input_ids = poi_inputs['input_ids'].squeeze()
        poi_attention_mask = poi_inputs['attention_mask'].squeeze()
        e3_input_ids = e3_inputs['input_ids'].squeeze()
        e3_attention_mask = e3_inputs['attention_mask'].squeeze()

        labels = torch.as_tensor(self.labels[index], dtype=torch.float)

        dataset = [d_input_ids, d_attention_mask, poi_input_ids, poi_attention_mask, e3_input_ids, e3_attention_mask, labels]
        return dataset


class ProtacDataModule(pl.LightningDataModule):
    def __init__(self, task_name, drug_model_name, poi_model_name, e3_model_name, num_workers, batch_size, prot_maxLength=545, traindata_rate = 1.0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_name = task_name
        self.prot_maxLength = prot_maxLength
        self.traindata_rate = traindata_rate
        
        self.d_tokenizer = AutoTokenizer.from_pretrained(drug_model_name)
        self.poi_tokenizer = AutoTokenizer.from_pretrained(poi_model_name)
        self.e3_tokenizer = AutoTokenizer.from_pretrained(e3_model_name)
        
        self.df_train = None
        self.df_val = None
        self.df_test = None

        self.load_testData = True

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None



    def get_task(self, task_name):
        if task_name.lower() == 'protacdb':
            return '/home/jina/PROTAC_NLP/PROTAC-Prediction/dataset/protacdb'

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from
        # a single process in distributed settings.
        dataFolder = self.get_task(self.task_name)

        self.df_train = pd.read_csv(dataFolder + '/train.csv')
        self.df_val = pd.read_csv(dataFolder + '/val.csv')

        ## -- Data Lenght Rate apply -- ##
        traindata_length = int(len(self.df_train) * self.traindata_rate)
        validdata_length = int(len(self.df_val) * self.traindata_rate)

        self.df_train = self.df_train[:traindata_length]
        self.df_val = self.df_val[:validdata_length]

        if self.load_testData is True:
            self.df_test = pd.read_csv(dataFolder + '/test.csv')



    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ProtacDataset(self.df_train.index.values, self.df_train.Label.values, self.df_train,
                                                  self.d_tokenizer, self.poi_tokenizer, self.e3_tokenizer, self.prot_maxLength)
            self.valid_dataset = ProtacDataset(self.df_val.index.values, self.df_val.Label.values, self.df_val,
                                                  self.d_tokenizer, self.poi_tokenizer, self.e3_tokenizer, self.prot_maxLength)

        if self.load_testData is True:
            self.test_dataset = ProtacDataset(self.df_test.index.values, self.df_test.Label.values, self.df_test,
                                                self.d_tokenizer, self.poi_tokenizer, self.e3_tokenizer, self.prot_maxLength)

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class ProtacModel(pl.LightningModule):
    def __init__(self, drug_model_name, poi_model_name, e3_model_name, lr, dropout, layer_features, loss_fn = "smooth", layer_limit = True, d_pretrained=True, poi_pretrained=True, e3_pretrained=True):
        super().__init__()
        self.lr = lr
        self.loss_fn = loss_fn
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion_smooth = torch.nn.SmoothL1Loss()

        #-- Pretrained Model Setting
        drug_config = AutoConfig.from_pretrained(drug_model_name)
        if d_pretrained is False:
            self.d_model = RobertaModel(drug_config)
        else:
            self.d_model = RobertaModel.from_pretrained(drug_model_name, num_labels=2,
                                                        output_hidden_states=True,
                                                        output_attentions=True)

        poi_config = AutoConfig.from_pretrained(poi_model_name)

        if poi_pretrained is False:
            self.poi_model = BertModel(poi_config)
        else:
            self.poi_model = BertModel.from_pretrained(poi_model_name,
                                                        output_hidden_states=True,
                                                        output_attentions=True)
        
        if layer_limit is True:
            self.poi_model = deleteEncodingLayers(self.poi_model, 18)
            
        e3_config = AutoConfig.from_pretrained(e3_model_name)

        if e3_pretrained is False:
            self.e3_model = BertModel(e3_config)
        else:
            self.e3_model = BertModel.from_pretrained(e3_model_name,
                                                        output_hidden_states=True,
                                                        output_attentions=True)

        if layer_limit is True:
            self.e3_model = deleteEncodingLayers(self.e3_model, 18)
            
        # TODO! 잘 체크하기!
        #-- Decoder Layer Setting
        layers = []
        firstfeature = self.d_model.config.hidden_size + self.poi_model.config.hidden_size + self.e3_model.config.hidden_size
        for feature_idx in range(0, len(layer_features) - 1):
            layers.append(nn.Linear(firstfeature, layer_features[feature_idx]))
            firstfeature = layer_features[feature_idx]

            if feature_idx is len(layer_features)-2:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    
        layers.append(nn.Linear(firstfeature, layer_features[-1]))
        self.decoder = nn.Sequential(*layers)

        self.save_hyperparameters()


    def forward(self, drug_inputs, prot_inputs):
        d_outputs = self.d_model(drug_inputs['input_ids'], drug_inputs['attention_mask'])
        poi_outputs = self.poi_model(poi_inputs['input_ids'], poi_inputs['attention_mask'])
        e3_outputs = self.e3_model(e3_inputs['input_ids'], e3_inputs['attention_mask'])

        outs = torch.cat((d_outputs.last_hidden_state[:, 0], poi_outputs.last_hidden_state[:, 0], e3_outputs.last_hidden_state[:, 0]), dim=1)
        outs = self.decoder(outs)        

        return outs

    def attention_output(self, drug_inputs, poi_inputs, e3_inputs):
        d_outputs = self.d_model(drug_inputs['input_ids'], drug_inputs['attention_mask'])
        poi_outputs = self.poi_model(poi_inputs['input_ids'], poi_inputs['attention_mask'])
        e3_outputs = self.e3_model(e3_inputs['input_ids'], e3_inputs['attention_mask'])

        outs = torch.cat((d_outputs.last_hidden_state[:, 0], poi_outputs.last_hidden_state[:, 0], e3_outputs.last_hidden_state[:, 0]), dim=1)
        outs = self.decoder(outs)        

        return d_outputs['attentions'], poi_outputs['attentions'], e3_outputs['attentions'], outs

    def training_step(self, batch, batch_idx):
        drug_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        poi_inputs = {'input_ids': batch[2], 'attention_mask': batch[3]}
        e3_inputs = {'input_ids': batch[4], 'attention_mask': batch[5]}
        labels = batch[6]

        output = self(drug_inputs, poi_inputs, e3_inputs)

        logits = output.squeeze(dim=1)
        
        if self.loss_fn == 'BCE':
            loss = self.criterion(logits, labels)
        else:
            loss = self.criterion_smooth(logits, labels)

        self.log("train_loss", loss)
        return {"loss": loss}

    # def training_step_end(self, outputs):
    #     return {"loss": outputs["loss"]}

    def validation_step(self, batch, batch_idx):
        drug_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        poi_inputs = {'input_ids': batch[2], 'attention_mask': batch[3]}
        e3_inputs = {'input_ids': batch[4], 'attention_mask': batch[5]}
        labels = batch[6]

        output = self(drug_inputs, poi_inputs, e3_inputs)

        logits = output.squeeze(dim=1)

        if self.loss_fn == 'BCE':
            loss = self.criterion(logits, labels)
        else:
            loss = self.criterion_smooth(logits, labels)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"logits": logits, "labels": labels}

    def validation_step_end(self, outputs):
        return {"logits": outputs['logits'], "labels": outputs['labels']}

    def validation_epoch_end(self, outputs):
        preds = self.convert_outputs_to_preds(outputs)
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0), dtype=torch.int)

        auroc, auprc, sensitivity, specificity, threshold = self.log_score(preds, labels)
                   
        self.log("valid_auroc", auroc, on_step=False, on_epoch=True, logger=True)
        self.log("valid_auprc", auprc, on_step=False, on_epoch=True, logger=True)

        self.log("valid_sens", sensitivity, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        drug_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        poi_inputs = {'input_ids': batch[2], 'attention_mask': batch[3]}
        e3_inputs = {'input_ids': batch[4], 'attention_mask': batch[5]}
        labels = batch[6]

        output = self(drug_inputs, poi_inputs, e3_inputs)
        logits = output.squeeze(dim=1)

        if self.loss_fn == 'BCE':
            loss = self.criterion(logits, labels)
        else:
            loss = self.criterion_smooth(logits, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"logits": logits, "labels": labels}

    def test_step_end(self, outputs):
        return {"logits": outputs['logits'], "labels": outputs['labels']}

    def test_epoch_end(self, outputs):
        preds = self.convert_outputs_to_preds(outputs)
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0), dtype=torch.int)

        auroc, auprc, sensitivity, specificity, threshold = self.log_score(preds, labels)

        self.log("test_auroc", auroc, on_step=False, on_epoch=True, logger=True)
        self.log("test_auprc", auprc, on_step=False, on_epoch=True, logger=True)

        self.log("test_sens", sensitivity, on_step=False, on_epoch=True, logger=True)
        self.log("test_spec", specificity, on_step=False, on_epoch=True, logger=True)
        self.log("test_thres", threshold, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.01
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
        )
        return optimizer

    def convert_outputs_to_preds(self, outputs):
        logits = torch.cat([output['logits'] for output in outputs], dim=0)
        return logits

    def log_score(self, preds, labels):
        y_pred = preds.detach().cpu().numpy()
        y_label = labels.detach().cpu().numpy()

        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        fpr = np.nan_to_num(fpr, nan=0.00001)

        precision = tpr / (tpr + fpr)
        precision = np.nan_to_num(precision, nan=0.00001)

        f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

        thred_optim = thresholds[np.argmax(f1)]

        print("optimal threshold: " + str(thred_optim))

        y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

        roc_score = auc(fpr, tpr)
        auroc_score = roc_auc_score(y_label, y_pred)
        auprc_score = average_precision_score(y_label, y_pred)

        cm1 = sk.metrics.confusion_matrix(y_label, y_pred_s)

        #####from confusion matrix calculate accuracy
        total1 = sum(sum(cm1))
        try:
            acc_score = (cm1[0, 0] + cm1[1, 1]) / total1
        except:
            acc_score = 0.0

        try:
            sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        except:
            sensitivity1 = 0.0

        try:
            specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        except:
            specificity1 = 0.0

        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

        try:
            metric_f1 = f1_score(y_label, outputs)
        except:
            metric_f1 = 0.0

        metric_recall = recall_score(y_label, y_pred_s)
        metric_precision =  precision_score(y_label, y_pred_s)
        
        print('\nConfusion Matrix : \n', cm1)
        print(f'\nRecall : {metric_recall}')  
        print(f'Precision : {metric_precision}')

        print("AUROC:", str(auroc_score))
        print("AUPRC:", str(auprc_score))
        
        print(f'accuracy : {acc_score}')
        print(f'F1 : {metric_f1}')

        return auroc_score, auprc_score, sensitivity1, specificity1, thred_optim

def wandb_train(config=None):
    gc.collect()
    torch.cuda.empty_cache()

    try:
        if config is not None:
            wandb.init(config=config, project=wandb_project)
        else:
            wandb.init()
        
        #밑에 두 줄 추가해봄
        config = load_hparams('/home/jina/PROTAC_NLP/PROTAC-Prediction/config/config_hparam.json')
        config = DictX(config)
        
        #config = wandb.config
        pl.seed_everything(seed=config.num_seed)

        d_model_name = "seyonec/PubChem10M_SMILES_BPE_450k"
        poi_model_name = "Rostlab/prot_bert_bfd"
        e3_model_name = "Rostlab/prot_bert_bfd"

        # 변수 dm에서 UnboundLocalError 발생!
        dm = ProtacDataModule(config.task_name, d_model_name, poi_model_name, e3_model_name,
                                 config.num_workers, config.batch_size, config.prot_maxlength, config.traindata_rate)
        dm.prepare_data()
        dm.setup()


        if config.model_mode == "test":
            #TODO! checkpoint의 경로설정
            # 체크포인트는 기본적으로 현재 dir내의 lightning_logs/version_LightningModule의 버전 번호/checkpoints 폴더에 저장됨
            # 실행이 안되니까 checkpoint 자체가 저장이 안되는 것 같음.
            #model = ProtacModel.load_from_checkpoint("checkpoint/bindingDB/epoch=33-step=13463.ckpt")
            print("wait____")
            
        else:
            model = ProtacModel(d_model_name, poi_model_name, e3_model_name,
                               config.lr, config.dropout, config.layer_features, config.loss_fn, config.layer_limit, config.pretrained['chem'], config.pretrained['poi'], config.pretrained['e3'])

        wandb_logger = WandbLogger(project=wandb_project)
        checkpoint_callback = ModelCheckpoint(monitor="valid_auroc", mode="max")
        # early_stop_callback = EarlyStopping(monitor="valid_auroc", patience=10, verbose=True, mode="max")
    
        trainer = pl.Trainer(gpus=config.gpu_ids,
                             max_epochs=config.max_epoch,
                             precision=16,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback],
                             accelerator='dp'
                             )

        if config.model_mode != "test":
            model.train()
            trainer.fit(model, datamodule=dm)

        ##-- Model Test --##
        if dm.load_testData is True:
            model.eval()
            trainer.test(model, datamodule=dm)


    except Exception as e:
        print(e)
        
    # UnboundLocalError: local variable 'dm' referenced before assignment
    # 해당 변수가 함수 내에서 선언되었지만 함수가 실행되는 도중에 해당 변수에 값을 할당하지 않은 경우
    del dm
    del model
    del wandb_logger
    del trainer

    gc.collect()
# ======== 

if __name__ == '__main__':
    wandb_project = "protac_project"

    ##-- hyper param config file Load --##
    config = load_hparams('/home/jina/PROTAC_NLP/PROTAC-Prediction/config/config_hparam.json')
    wandb_train(config)

    
#     # #-- wandb Sweep Hyper Param Tuning --##
#     # config = load_hparams('/home/jina/PROTAC_NLP/PROTAC-Prediction/config/config_sweep.json')
#     # sweep_id = wandb.sweep(config, project=wandb_project)
#     # wandb.agent(sweep_id, wandb_train)
    
    