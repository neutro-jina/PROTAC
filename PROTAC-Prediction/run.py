import os
import pandas as pd

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

# from utils.utils import *
from util import *

from tqdm import tqdm
from train import ProtacModel

def get_protac(drug_inputs, poi_inputs, e3_inputs):
    output_preds = model(drug_inputs, poi_inputs, e3_inputs)

    m = torch.nn.Sigmoid()
    predict = torch.squeeze(m(output_preds)).tolist()

    # output_preds = torch.relu(output_preds)
    # predict = torch.tanh(output_preds)
    # predict = predict.squeeze(dim=1).tolist()

    return predict


if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    device_count = torch.cuda.device_count()
    device_protac = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    device = torch.device('cpu')

    # ------------
    # model
    # ------------
    d_model_name = "seyonec/PubChem10M_SMILES_BPE_450k"
    poi_model_name = "Rostlab/prot_bert_bfd"
    e3_model_name = "Rostlab/prot_bert_bfd"

    tokenizer = AutoTokenizer.from_pretrained(d_model_name)
    poi_tokenizer = AutoTokenizer.from_pretrained(poi_model_name)
    e3_tokenizer = AutoTokenizer.from_pretrained(e3_model_name)

    #--protac Model
    ##-- hyper param config file Load --##
    config = load_hparams('/home/jina/PROTAC_NLP/PROTAC-Prediction/config/config_hparam.json')
    config = DictX(config)
    model = ProtacModel(d_model_name, poi_model_name, e3_model_name,
                               config.lr, config.dropout, config.layer_features, config.loss_fn, config.layer_limit, config.pretrained['chem'], config.pretrained['prot']).to(device_protac)

    # model = ProtacModel.load_from_checkpoint('./biomarker_bindingdb_train8595_pretopre/3477h3wf/checkpoints/epoch=30-step=7284.ckpt').to(device_protac)

    model.eval()
    model.freeze()
    
    if device_protac.type == 'cuda':
        model = torch.nn.DataParallel(model)
                             
    # smiles_aas_test()
    
    
    
print("done!")

