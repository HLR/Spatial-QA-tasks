import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
from tqdm import tqdm
# from torchvision import models
# from transformers import AdamW
from BertModels import BertForMultipleClass, BertForBooleanQuestionYN ,BertForBooleanQuestionFR, BertForQuestionAnswering,  BertForBooleanQuestionFR1, BertForBooleanQuestionFB,BertForBooleanQuestionFB1,  BertForBooleanQuestionYNboolq ,BertForBooleanQuestionYN1 , BertForBooleanQuestionCO, BertForBooleanQuestionCO1, BertForMaskedLM, BertForTokenClassification, BertForBooleanQuestion3ClassYN,  BertForMultipleClassLoad, BertForSequenceClassification, BertForSequenceClassification1, BertForSequenceClassification2, BertForSequenceClassification3, BertForBooleanQuestionYNsprlqa, BertForBooleanQuestionYNsprlqaLoad
# from BertModels import BertForBooleanQuestionYN ,BertForBooleanQuestionYN1
from PLModels import  BertMultiTaskBooleanQuestion, BertMultiTaskMultipleClass, BertMultiTaskBooleanQuestionLoad, BertMultiTaskMultipleClassLoad, BertForSpatialRelationExtraction #, MultipleClass, BooleanQuestionYN ,BooleanQuestionFR#, MultipleClassLoad,  BooleanQuestionLoad,
from XLNETModels import XLNETForQuestionAnswering, XLNETForBooleanQuestionFR, XLNETForBooleanQuestionFB, XLNETForBooleanQuestionYN , XLNETForBooleanQuestionCO
from ALBertModels import ALBertForQuestionAnswering, ALBertForBooleanQuestionFR, ALBertForBooleanQuestionFB, ALBertForBooleanQuestionYN , ALBertForBooleanQuestionCO
from BertSpatialQA import SpatialQA, SpatialQAaddSprl, SpatialQAaddSprlLoad,SpatialQAaddSprlTriplet, SpatialQASupervised, weights_init_normal
from Create_LM_input_output import initialize_tokenizer
from consistency_check import consistency, contrast
import matplotlib.pyplot as plt

#adding arguments
parser = argparse.ArgumentParser()

parser.add_argument("--research_hlr",help="change the location of files",action='store_true', default = True)
parser.add_argument("--result",help="Name of the result's saving file", type= str, default='test')
parser.add_argument("--result_folder",help="Name of the folder of the results file", type= str, default='transfer/Results')
parser.add_argument("--model",help="Name of the model's saving file", type= str, default='')
parser.add_argument("--model_folder",help="Name of the folder of the models file", type=str, default = "transfer/Models")

parser.add_argument("--old_experiments",help="from the spartun project some setting of models changes, so if you want to run the previous things, set this True", default = False, action='store_true')

parser.add_argument("--dataset",help="name of the dataset like mSpRL or spaceeval", type = str, default = 'spartqa')

parser.add_argument("--no_save",help="If save the model or not", action='store_true', default = False)
parser.add_argument("--load",help="For loading model", type=str)
parser.add_argument("--cuda",help="The index of cuda", type=int, default=None)
parser.add_argument("--qtype",help="Name of Question type. (FB, FR, CO, YN)", type=str, default = 'all')
parser.add_argument("--train10k",help="Train on 10k data for babi dataset", action='store_true', default = False)
parser.add_argument("--train1k",help="Train on 1k data for babi dataset", action='store_true', default = False)
parser.add_argument("--train24k",help="Train on 24k data", action='store_true', default = False)
parser.add_argument("--train100k",help="Train on 100k data", action='store_true', default = False)
parser.add_argument("--train500",help="Train on 500 data", action='store_true', default = False)
parser.add_argument("--unseentest",help="Test on unseen data", action='store_true', default = False)
parser.add_argument("--human",help="Train and Test on human data", action='store_true', default = False)
parser.add_argument("--humantest",help="Test on human data", action='store_true', default = False)
parser.add_argument("--dev_exists", help="If development set is used", action='store_true', default = False)
parser.add_argument("--test_track", help="track the test result during training", action='store_true', default = False)
parser.add_argument("--no_train",help="Number of train samples", action='store_true', default = False)
parser.add_argument("--save_data",help="save extracted data", action='store_true', default = False)

parser.add_argument("--baseline",help="Name of the baselines. Options are 'bert', 'xlnet', 'albert'", type=str, default = 'bert')
parser.add_argument("--pretrain",help="Name of the pretrained model. Options are 'bertqa', 'bertbc' (for bert boolean clasification), 'mlm', 'mlmr', 'tokencls'", type=str, default = 'bertbc')
parser.add_argument("--con",help="Testing consistency or contrast", type=str, default = 'not')
parser.add_argument("--optim",help="Type of optimizer. options 'sgd', 'adamw'.", type=str, default = 'adamw')
parser.add_argument("--loss",help="Type of loss function. options 'cross'.", type=str, default = 'focal')
parser.add_argument("--batch_size",help="size of batch. If none choose the whole example in one sample. If QA number of all questions if SIE number of sentences or triplets'.", type=int, default = 1)
parser.add_argument("--best_model",help="How to save the best model. based on aacuracy or f1 measure", type=str, default = 'accuracy')

parser.add_argument("--train",help="Number of train samples", type = int)
parser.add_argument("--train_log", help="save the log of train if true", default = False, action='store_true')
parser.add_argument("--start",help="The start number of train samples", type = int, default = 0)
parser.add_argument("--dev",help="Number of dev samples", type = int)
parser.add_argument("--test",help="Number of test samples", type = int)
parser.add_argument("--unseen",help="Number of unseen test samples", type = int)
parser.add_argument("--has_zero_eval", help="If True before starting the training have a test on the test set", default = False, action='store_true')

parser.add_argument("--stepgame_train_set",help="Number of sentence in stepgame dataset", type = str, default=None)
# parser.add_argument("--stepgame_dev_sets",help="Number of sentence in stepgame dataset", type = list, default=[12345])
parser.add_argument("--stepgame_test_set",help="Number of sentence in stepgame dataset", type = str, default="1 2 3 4 5 6 7 8 9 10")

parser.add_argument("--epochs",help="Number of epochs for training", type = int, default=0)
parser.add_argument("--lr",help="learning rate", type = float, default=2e-6)

parser.add_argument("--dropout", help="If you want to set dropout=0", action='store_true', default = False)
parser.add_argument("--unfreeze", help="freeze the first layeres of the model except this numbers", type=int, default = 0)
parser.add_argument("--seed", help="set seed for reproducible result", type=int, default = 1)

parser.add_argument("--other_var",  dest='other_var', action='store', help="Other variable: classification (DK, noDK), random, fine-tune on unseen. for changing model load MLM from pre-trained model and replace other parts with new on", type=str)
parser.add_argument("--other_var2",  dest='other_var2', action='store', help="Other variable: classification (DK, noDK), random, fine-tune on unseen. for changing model load MLM from pre-trained model and replace other parts with new on", type=str)
parser.add_argument("--detail",help="a description about the model", type = str)

#arguments for end2end models
parser.add_argument("--options", help="describe the model features: 'q+s' + 'first_attention_stoq' + 'just_pass_entity'+ '2nd_attention_stoq'+ '2nd_attention_qtos' + ", type=str, default=None)
parser.add_argument("--top_k_sent", help="set top k for sentence", type=int, default=None)
parser.add_argument("--top_k_s", help="set top k for indicator, entity, and triplets: 3#4#3", type=str, default=None)
parser.add_argument("--top_k_q", help="set top k for indicator, entity, and triplets:  3#4#3", type=str, default=None)
parser.add_argument("--cls_input_dim", help="an integer based on the final input of boolean classification", type=int, default=768)

args = parser.parse_args()

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# print(os.getcwd())

start_path = '/egr/research-hlr/' #'/tank/space/' #'/egr/research-hlr/' #if args.research_hlr else '/tank/space/'

if args.pretrain in ["tokencls", "sptypecls", "spcls", "sre"]:
    result_adress = os.path.join(start_path+'rshnk/'+args.result_folder+ '/'+args.dataset+'/'+args.baseline+'_SIE/',args.result)

else:
    result_adress = os.path.join(start_path+'rshnk/'+args.result_folder+ '/'+args.dataset+'/'+args.baseline+'/',args.result)

model_address = os.path.join(start_path+'rshnk/'+args.model_folder, args.dataset)

args.stepgame_test_set = [int(i) for i in args.stepgame_test_set.split(' ')]

if not os.path.exists(result_adress):
    os.makedirs(result_adress)
if not os.path.exists(model_address):
    os.makedirs(model_address)

#saved_file = open('results/train'+args.result+'.txt','w')
#choosing device
if torch.cuda.is_available():
    print('Using ', torch.cuda.device_count() ,' GPU(s)')
    mode = 'cuda:'+str(args.cuda) if args.cuda else 'cuda'    
    if args.seed:  
        torch.cuda.manual_seed(args.seed)

else:
    print("WARNING: No GPU found. Using CPUs...")
    mode = 'cpu'
    
device = torch.device(mode)

if args.seed:
    print("set seeds.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def config():

    f = open(result_adress+'/config.txt','w')
    print('Configurations:\n', args , file=f)
    f.close()

config()

epochs = args.epochs
if args.human: args.humantest = True

if args.train24k: train_num = 'train24k'
elif args.train100k: train_num = 'train100k'
elif args.train500: train_num = 'train500'
elif args.train1k: train_num = 'train1k'
elif args.train10k: train_num = 'train10k'
else: train_num = None

if args.model == '': args.model = args.result

if args.baseline == "roberta": pretrained_data = 'roberta-base'
elif args.baseline == "xlnet": pretrained_data = 'xlnet-base-cased'
elif args.baseline == "albert": pretrained_data = 'albert-base-v2'
else: pretrained_data = 'bert-base-uncased'

initialize_tokenizer(args.baseline, pretrained_data)

num_labels_YN = None
num_labels_FR = None
#calling test and train based on the task
if args.pretrain == 'tokencls':
    if args.dataset == 'msprl':
        from spInfo.msprl.train_tokencls_msprl import train
        from spInfo.msprl.test_tokencls_msprl import test
    elif args.dataset == 'spaceEval':
        from spaceeval.train_tokencls_spaceEval import train
        from spaceeval.test_tokencls_spaceEval import test
    elif args.dataset == "stepgame":
        from spInfo.stepgame.test_tokencls import test
    else:
        from spInfo.train_tokencls import train
        from spInfo.test_tokencls import test
elif args.pretrain == "sre":
    if args.dataset == "spartqa":
        from SRE.train import train
        from SRE.test import test
        sre_num_labels = 12
elif args.pretrain == 'spcls' or args.pretrain == 'sptypecls':
    if args.dataset == 'msprl':
        from spInfo.msprl.train_spcls_msprl import train
        from spInfo.msprl.test_spcls_msprl import test
    elif args.dataset == "stepgame":
        from spInfo.stepgame.test_spcls import test
    else:    
#         if args.humantest:
#             from spInfo.test_spcls_no_annot import test
#         else:
        from spInfo.train_spcls import train
        from spInfo.test_spcls import test

elif args.pretrain == 'end2end':
    if args.dataset == 'stepgame':
        if args.other_var == 'addsprl':
            from end2end.StepGame.train import train
            from end2end.StepGame.test import test
    elif args.dataset == 'sprlqa':
        if args.other_var == 'addsprl':
            from end2end.sprlqa.train import train
            from end2end.sprlqa.test import test
    else:
        if args.other_var == 'supervised':
            from end2end.train_sup import train
            from end2end.test_sup import test
        else:
            from end2end.train import train
            from end2end.test import test
        
    
elif args.pretrain == 'sptype+bertbc':
    from QA_splinfo.train import train
    from QA_splinfo.test import test
    
elif args.pretrain == 'sptypeQA':
    from QA_splinfo.train_triplet import train
    from QA_splinfo.test_triplet import test
    
else: #QA task
    
    if args.dataset == 'boolq':   
        from boolq.train_boolQ import train
        from boolq.test_boolQ import test
        
        num_labels_YN = 2
        
    elif args.dataset == 'babi':
        from QA.babi.train import train
        from QA.babi.test import test
        if args.qtype in ["all", "YN"]:
            num_labels_YN = 2
        if args.qtype in ["all", "FR"]:
            num_labels_FR = 4
    
    elif args.dataset == 'sprlqa':
        # from msprl.QA.train import train 
        # from msprl.QA.test import test
        from QA.sprlqa.train import train
        from QA.sprlqa.test import test 
        num_labels_YN = 2

    elif args.dataset == 'stepgame':
        from QA.StepGame.train import train
        from QA.StepGame.test import test    
        num_labels_FR = 9
    else:
        if args.old_experiments:
            from QA.trainold import train
            from QA.testold import test
        else:
            from QA.train import train
            from QA.test import test
        if args.dataset == "spartqa":
            if args.qtype in ["all", "YN"]:
                num_labels_YN = 3
            if args.qtype in ["all", "FR"]:
                num_labels_FR = 7
        else: #spartun
            if args.qtype in ["all", "YN"]:
                num_labels_YN = 2
            if args.qtype in ["all", "FR"]:
                num_labels_FR = 15
#model
# model = None
if args.load:
    
#     print('/tank/space/rshnk/'+args.model_folder+'/'+args.load+'.th')
    model = torch.load(start_path+'rshnk/'+args.model_folder+'/'+args.load+'.th', map_location={'cuda:0': 'cuda:'+str(args.cuda),'cuda:1': 'cuda:'+str(args.cuda),'cuda:2': 'cuda:'+str(args.cuda),'cuda:3': 'cuda:'+str(args.cuda), 'cuda:5': 'cuda:'+str(args.cuda), 'cuda:4': 'cuda:'+str(args.cuda), 'cuda:6': 'cuda:'+str(args.cuda),'cuda:7': 'cuda:'+str(args.cuda)})
#     model.to(device)
    
    if args.unfreeze:
        if args.baseline == 'bert':
            for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)]: 
        #         print('I will be frozen: {}'.format(name)) 
                param.requires_grad = False
    
    if args.other_var == 'change_model' or args.other_var2 == 'change_model':
        
        pretrained_dict = model.state_dict()
        if args.pretrain == 'bertbc':
            if args.old_experiments:
                if args.qtype == 'YN':
                
                    if args.baseline == 'bert':
                        if args.dataset == 'spartqa':
                            model2 = BertForBooleanQuestionYN1.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
                        elif args.dataset == 'sprlqa':
                            model2 = BertForBooleanQuestionYNsprlqaLoad.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
                
                elif args.qtype == 'FB':
                    if args.baseline == 'bert':
                        model2 = BertForBooleanQuestionFB1.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
                elif args.qtype == 'FR':
                    if args.baseline == 'bert':
                        model2 = BertForBooleanQuestionFR1.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
                elif args.qtype == 'CO':
                    if args.baseline == 'bert':
                        model2 = BertForBooleanQuestionCO1.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
            else:
                if args.baseline == 'bert':
                    model2 = BertMultiTaskBooleanQuestionLoad.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout, num_labels_YN = num_labels_YN, num_labels_FR= num_labels_FR,  dataset = "human" if args.human else args.dataset, LM = args.baseline, has_batch = True if args.batch_size and args.batch_size>1 else False, criterion = args.loss)
                    
        elif args.pretrain == 'bertmc':
            if args.old_experiments:
                if args.qtype == 'YN':
                    if args.baseline == 'bert':
                        model2 =  BertForMultipleClassLoad.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout)
                if args.qtype == 'FR':
                    if args.baseline == 'bert':
                        model2 =  BertForMultipleClassLoad.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout, qtype = 'FR', num_classes = 9)
            else:
                if args.baseline == 'bert':
                    model2 = BertMultiTaskMultipleClassLoad.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout, num_classes_YN = num_labels_YN , num_classes_FR=  num_labels_FR, dataset = "human" if args.human else args.dataset,  LM = args.baseline , has_batch = True if args.batch_size and args.batch_size>1 else False, criterion = args.loss)
                
        elif args.pretrain == 'sptypecls':
            if args.human:
                model2 = BertForSequenceClassification2.from_pretrained(pretrained_data, num_labels = 1, type_class = 11 , device = device,  no_dropout= args.dropout) 
            if args.baseline == 'bert' and args.dataset == 'msprl':
                model2 = BertForSequenceClassification2.from_pretrained(pretrained_data, num_labels = 1, type_class = 23 , device = device,  no_dropout= args.dropout)
        elif args.pretrain == 'end2end':
            if args.qtype == 'YN': qa_num_labels = 2
            elif args.qtype == 'FR': qa_num_labels = 7
            elif args.qtype == 'CO': qa_num_labels = 2
            elif args.qtype == 'FB': qa_num_labels = 3
            else: qa_num_labels = None
            if args.baseline == 'bert':
                drop = 0 if args.dropout else 0.1
                if args.other_var == 'addsprl':
                    # model2 = SpatialQAaddSprl(no_dropout=drop, qa_num_labels = qa_num_labels, rel_type_num = 11, qtype = args.qtype, device = device, unfreeze = args.unfreeze, top_k_s= args.top_k_s.split('#') if args.top_k_s else None, top_k_q= args.top_k_q.split('#') if args.top_k_q else None, options= args.options, cls_input_dim = args.cls_input_dim)
                    model2 = SpatialQAaddSprlLoad(no_dropout=drop, qa_num_labels = qa_num_labels, rel_type_num = 11, qtype = args.qtype, device = device, unfreeze = args.unfreeze, top_k_sent= args.top_k_sent, top_k_s= args.top_k_s.split('#') if args.top_k_s else None, top_k_q= args.top_k_q.split('#') if args.top_k_q else None, options= args.options, cls_input_dim = args.cls_input_dim)
                
        if args.baseline == 'bert':
            if args.unfreeze:
                    for name, param in list(model2.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                        #print('I will be frozen: {}'.format(name)) 
                        param.requires_grad = False
        
        model_dict = model2.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print(pretrained_dict.keys())
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # # 3. load the new state dict
        model2.load_state_dict(model_dict)
        
        model = model2
        
    model.to(device)   
else:
    if args.pretrain == 'bertqa': # for FA
        if args.baseline == 'bert':
            model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        elif args.baseline == 'albert':
            model = ALBertForQuestionAnswering.from_pretrained(pretrained_data,  device = device)
        elif args.baseline == 'xlnet':
            model = XLNETForQuestionAnswering.from_pretrained(pretrained_data,  device = device)
        model.to(device)
    
    elif args.pretrain == 'mlm' or args.pretrain =='mlmr':
        if args.baseline == 'bert':
            drop = 0 if args.dropout else 0.1
            #bert-large-uncased-whole-word-masking-finetuned-squad
#             bert-base-uncased
            model = BertForMaskedLM.from_pretrained(pretrained_data, hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True)
    
            if args.unfreeze:
                for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                    #print('I will be frozen: {}'.format(name)) 
                    param.requires_grad = False
        model.to(device)
    
    elif args.pretrain == 'end2end':
        if args.qtype == 'YN': 
            if args.dataset == 'sprlqa': qa_num_labels = 2
            else: qa_num_labels = 3
        elif args.qtype == 'FR': 
            if args.dataset != "stepgame": qa_num_labels = 7
            else: qa_num_labels = 9
        elif args.qtype == 'CO': qa_num_labels = 2
        elif args.qtype == 'FB': qa_num_labels = 3
        else: qa_num_labels = None
        
        if args.baseline == 'bert':
            drop = 0 if args.dropout else 0.1
            
            if args.other_var == 'addsprl':
                
#                 model = SpatialQAaddSprl(no_dropout=drop, qa_num_labels = qa_num_labels, rel_type_num = 11, qtype = args.qtype, device = device, unfreeze = args.unfreeze, top_k_s= args.top_k_s.split('#') if args.top_k_s else None, top_k_q= args.top_k_q.split('#') if args.top_k_q else None, options= args.options, cls_input_dim = args.cls_input_dim)
                model = SpatialQAaddSprl(no_dropout=drop, qa_num_labels = qa_num_labels, rel_type_num = 11, qtype = args.qtype, device = device, unfreeze = args.unfreeze, top_k_sent= args.top_k_sent, top_k_s= args.top_k_s.split('#') if args.top_k_s else None, top_k_q= args.top_k_q.split('#') if args.top_k_q else None, options= args.options, cls_input_dim = args.cls_input_dim)

            elif args.other_var == 'supervised':
                model = SpatialQASupervised(no_dropout=drop, qa_num_labels = qa_num_labels, rel_type_num = 11, qtype = args.qtype, device = device, unfreeze = args.unfreeze)
                
            else:
                model = SpatialQA(no_dropout=drop, qa_num_labels = qa_num_labels, rel_type_num = 11, qtype = args.qtype, device = device, unfreeze = args.unfreeze)
        
        model.to(device)
    
    elif args.pretrain == "sre":
        model = BertForSpatialRelationExtraction( no_dropout = drop, num_labels = sre_num_labels, device =device)
        model.to(device)
    
    elif args.pretrain == 'sptypeQA':
        
        if args.qtype == 'YN': qa_num_labels = 2
        elif args.qtype == 'FR': qa_num_labels = 7
        elif args.qtype == 'CO': qa_num_labels = 2
        elif args.qtype == 'FB': qa_num_labels = 3
        else: qa_num_labels = None
        
        if args.baseline == 'bert':
            drop = 0 if args.dropout else 0.1
            model = SpatialQAaddSprlTriplet(no_dropout=drop, qa_num_labels = qa_num_labels, rel_type_num = 11, qtype = args.qtype, device = device, unfreeze = args.unfreeze)
            model.to(device)
            
    elif args.pretrain == 'tokencls':
        if args.baseline == 'bert':
            drop = 0 if args.dropout else 0.1
            model = BertForTokenClassification.from_pretrained(pretrained_data, hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True, num_labels = 5)
            
            if args.unfreeze:
                for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                    #print('I will be frozen: {}'.format(name)) 
                    param.requires_grad = False    
        
        model.to(device)
    
    elif args.pretrain == 'spcls':
        if args.baseline == 'bert':
            drop = 0 if args.dropout else 0.1
            model = BertForSequenceClassification.from_pretrained(pretrained_data, num_labels = 1, device = device,  no_dropout= args.dropout)
#             model = BertForSequenceClassification.from_pretrained(pretrained_data, hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True, num_labels = 1)

            if args.unfreeze:
                for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                    #print('I will be frozen: {}'.format(name)) 
                    param.requires_grad = False
        model.to(device)
        
    elif args.pretrain == 'sptypecls':
        if args.baseline == 'bert':
#             drop = 0 if args.dropout else 0.1
            if args.dataset == 'msprl':
                model = BertForSequenceClassification3.from_pretrained(pretrained_data, num_labels = 1, type_class = 23 , device = device,  no_dropout= args.dropout)
                
            elif args.dataset == 'spaceEval':
                model = BertForSequenceClassification1.from_pretrained(pretrained_data, num_labels = 1, type_class = 22 , device = device,  no_dropout= args.dropout)
            elif args.dataset == 'spartun':
                model = BertForSequenceClassification3.from_pretrained(pretrained_data, num_labels = 1, type_class = 16 , device = device,  no_dropout= args.dropout)
            else:
                model = BertForSequenceClassification3.from_pretrained(pretrained_data, num_labels = 1, type_class = 11 , device = device,  no_dropout= args.dropout)
#             model = BertForSequenceClassification.from_pretrained(pretrained_data, hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True, num_labels = 1)

            #unfreeze the layers
            if args.unfreeze:
                for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                    #print('I will be frozen: {}'.format(name)) 
                    param.requires_grad = False
        model.to(device)
    
    elif args.pretrain == 'bertmc':
        if args.old_experiments:
            if args.qtype == 'YN':
    #             drop = 0 if args.dropout else 0.1
                if args.baseline == 'bert':
                    model =  BertForMultipleClass.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout)
    #                 model =  BertForMultipleChoice.from_pretrained(pretrained_data,  hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True)
                    if args.unfreeze:
                        for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                            #print('I will be frozen: {}'.format(name)) 
                            param.requires_grad = False    
            
            elif args.qtype == 'FR':
    #             drop = 0 if args.dropout else 0.1
                if args.baseline == 'bert':
                    if args.dataset == 'stepgame':
                        model =  BertForMultipleClass.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout, qtype = args.qtype, num_classes = 9)
        #                 model =  BertForMultipleChoice.from_pretrained(pretrained_data,  hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True)
                        if args.unfreeze:
                            for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                                #print('I will be frozen: {}'.format(name)) 
                                param.requires_grad = False  
        else:
            #using PLModels file

            model =  BertMultiTaskMultipleClass.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout, num_classes_YN = num_labels_YN , num_classes_FR=  num_labels_FR, dataset = "human" if args.human else args.dataset,  LM = args.baseline , has_batch = True if args.batch_size and args.batch_size>1 else False, criterion = args.loss)
#                 model =  BertForMultipleChoice.from_pretrained(pretrained_data,  hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True)
            if args.unfreeze:
                if args.baseline == "bert":
                    for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                        #print('I will be frozen: {}'.format(name)) 
                        param.requires_grad = False

        model.to(device)
    
    
    elif args.pretrain == 'bertbc' or args.pretrain == 'sptype+bertbc':
        
        if args.old_experiments:
            if args.qtype == 'FR':
                if args.baseline == 'bert':
                    if args.dataset == "babi":
                        model = BertForBooleanQuestionFR.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout, num_labels = 4)
                    else:
                        model = BertForBooleanQuestionFR.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout)
                    if args.unfreeze:
                        for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                            #print('I will be frozen: {}'.format(name)) 
                            param.requires_grad = False
                elif args.baseline == 'albert':
                    model = ALBertForBooleanQuestionFR.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout)
                elif args.baseline == 'xlnet':
                    model = XLNETForBooleanQuestionFR.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout)
                
            elif args.qtype == 'FB':
                if args.baseline == 'bert':
                    model = BertForBooleanQuestionFB.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout)
                    if args.unfreeze:
                        for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                            #print('I will be frozen: {}'.format(name)) 
                            param.requires_grad = False
                elif args.baseline == 'albert':
                    model = ALBertForBooleanQuestionFB.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout)
                elif args.baseline == 'xlnet':
                    model = XLNETForBooleanQuestionFB.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout)
                
            elif args.qtype == 'YN' and args.other_var == 'DK':
                
                if args.baseline == 'bert':
                    model =  BertForBooleanQuestion3ClassYN.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout) 
                    if args.unfreeze:
                        for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                            #print('I will be frozen: {}'.format(name)) 
                            param.requires_grad = False
            
    #         elif args.type =='YN' and args.other_var == 'YN1':
    #             if args.baseline == 'bert':
    #                 model =  BertForBooleanQuestionYN1.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)    
    #             model.to(device)
            elif args.qtype == 'YN' and args.dataset == 'boolq':
                if args.baseline == 'bert':
                    model = BertForBooleanQuestionYNboolq.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
                    if args.unfreeze:
                        for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                            #print('I will be frozen: {}'.format(name)) 
                            param.requires_grad = False
                
            elif args.qtype == 'YN' and (args.dataset == 'sprlqa' or args.dataset == 'spartun'):
                if args.baseline == 'bert':
                    model = BertForBooleanQuestionYNsprlqa.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
                    if args.unfreeze:
                        for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                            #print('I will be frozen: {}'.format(name)) 
                            param.requires_grad = False
            
            
            elif args.qtype == 'YN':
                if args.baseline == 'bert':
                    model = BertForBooleanQuestionYN.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
                    if args.unfreeze:
                        for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                            #print('I will be frozen: {}'.format(name)) 
                            param.requires_grad = False
                elif args.baseline == 'albert':
                    model = ALBertForBooleanQuestionYN.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
                elif args.baseline == 'xlnet':
                    model = XLNETForBooleanQuestionYN.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
            
            
                
            elif args.qtype == 'CO':
                if args.baseline == 'bert':
                    model = BertForBooleanQuestionCO.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
                    if args.unfreeze:
                        for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                            #print('I will be frozen: {}'.format(name)) 
                            param.requires_grad = False
                elif args.baseline == 'albert':
                    model = ALBertForBooleanQuestionCO.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
                elif args.baseline == 'xlnet':
                    model = XLNETForBooleanQuestionCO.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)

        else:
#             if args.qtype == "FR":
#                 model =  BooleanQuestionFR.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout, num_labels_FR= num_labels_FR,  dataset = "human" if args.human else args.dataset)
#             elif args.qtype == "YN":
#                 model =  BooleanQuestionYN.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout, num_labels_YN= num_labels_YN,  dataset = "human" if args.human else args.dataset)
            
            model =  BertMultiTaskBooleanQuestion.from_pretrained(pretrained_data,  device = device,  no_dropout= args.dropout, num_labels_YN = num_labels_YN, num_labels_FR= num_labels_FR,  dataset = "human" if args.human else args.dataset, LM = args.baseline, has_batch = True if args.batch_size and args.batch_size>1 else False, criterion = args.loss)
# #                 model =  BertForMultipleChoice.from_pretrained(pretrained_data,  hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True)
            if args.unfreeze:
                if args.baseline == "bert":
                    for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                        #print('I will be frozen: {}'.format(name)) 
                        param.requires_grad = False
        
        model.to(device)
                
    #     elif args.pretrain == 'sptype+bertbc':
    #         if args.qtype == 'YN':
    #             if args.baseline == 'bert':
    #                 model = BertForBooleanQuestionYN.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
    #                 if args.unfreeze:
    #                     for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
    #                         #print('I will be frozen: {}'.format(name)) 
    #                         param.requires_grad = False
    #             elif args.baseline == 'albert':
    #                 model = ALBertForBooleanQuestionYN.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
    #             elif args.baseline == 'xlnet':
    #                 model = XLNETForBooleanQuestionYN.from_pretrained(pretrained_data,  device = device, no_dropout= args.dropout)
    #             model.to(device)
            

# model

# optimizer = None
if args.optim == 'sgd': 
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
elif args.optim == 'adamw':
     optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
elif args.optim == 'adam':
     optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

criterion = None
# if args.loss == 'cross':
#     criterion = nn.CrossEntropyLoss()


#zero_evaluation of model before any training
if args.has_zero_eval: 
    zero_test_file = open(result_adress+'/zero_step_test.txt','w')
    
    test_accuracy = test(model
                          , pretrain = args.pretrain
                          , baseline = args.baseline
                          , test_or_dev = 'test'
                          , num_sample = args.test
                          , train_num = train_num
                          , unseen = False
                          , qtype = args.qtype
                          , other = args.other_var
                          , data_name =  ("human" if args.human else args.dataset)
                          , save_data = args.save_data
                          , device = device
                          , file = zero_test_file
                        ) 
    
#     test_all_accuracy.append(test_accuracy) 
    zero_test_file.close()



#training starts
all_loss, inter_test_all_accuracy, dev_all_accuracy, inter_test_unseen_all_accuracy, human_all_accuracy = [], [], [],[], []
all_f1, inter_test_all_f1, dev_all_f1, inter_test_unseen_all_f1, human_all_f1 = [], [], [],[], []
all_accuracy = []
best_val, best_val_unseen = -0.1, -0.1


if not args.no_train: 
    print('~~~~~~~~~~~~ Train ~~~~~~~~~~~~ ')
    train_file = open(result_adress+'/train.txt','w')
    inter_test_file = open(result_adress+'/intermediate_test.txt','w')
if args.dev_exists:
    dev_file = open(result_adress+'/dev.txt','w')

no_changes = 0
for ep in tqdm(range(epochs)):
    
    #train
    if args.no_train != True:
        print('******** Epoch '+str(ep)+' ******** ', file = train_file)
        losses, result = train(model
                               , criterion = criterion
                               , optimizer = optimizer
                               , pretrain = args.pretrain
                               , baseline = args.baseline
                               , start = args.start
                               , num_sample = args.train
                               , train_num =  (int(args.stepgame_train_set) if args.stepgame_train_set else None) if args.dataset == "stepgame" else train_num
                               , qtype = args.qtype
                               , data_name =  "human" if args.human else args.dataset
                               , other = args.other_var
                               , device = device
                               , train_log = args.train_log
                               , file = train_file
                               , batch_size = args.batch_size
                              )
        #result[0] == accuracy, result[1] if exists = f1

        all_loss.append(losses)
        if args.qtype == 'YN' or args.pretrain in ["tokencls", "sptypecls", "spcls"]:
            all_f1.append(result[1])
        
        all_accuracy.append(result[0])
    
    #save model
    if not args.no_save:
#         print('/tank/space/rshnk/'+args.model_folder+'/model_'+args.baseline+('' if args.dataset == 'spartqa' else '_'+args.dataset)+'_final_'+args.model+'.th')
        torch.save(model, model_address+'/model_'+args.baseline+'_'+args.dataset+'_final_'+args.model+'.th')
    
    if not args.dev_exists or args.test_track:
#         if args.human: 
        inter_test_result = test(model
                          , pretrain = args.pretrain
                          , baseline = args.baseline
                          , test_or_dev = 'test'
                          , num_sample = args.test
                          , train_num = args.stepgame_test_set[0] if args.dataset == "stepgame" else train_num
                          , unseen = False
                          , qtype = args.qtype
                          , other = args.other_var
                          , data_name =   "human" if args.human else args.dataset
                          , save_data = args.save_data
                          , device = device
                          , file = inter_test_file
                        ) 
#         else: 
#             inter_test_accuracy = test(model, args.pretrain, args.baseline, 'test', args.test, False, args.qtype, args.other_var, args.humantest, device, inter_test_file)
        if args.qtype == 'YN' or args.pretrain in ["tokencls", "sptypecls", "spcls"]:
            inter_test_all_f1.append(inter_test_result[1])
            f1 = inter_test_result[1]
        inter_test_all_accuracy.append(inter_test_result[0])
        accu = inter_test_result[0]

        # show image of accuracy  
        plt.figure()
        plt.plot(inter_test_all_accuracy, label="accuracy")
        plt.legend()
        plt.savefig(result_adress+'/inter_test_plot_acc.png')
    #     plt.show()
        plt.close()
        if args.qtype == 'YN' or args.pretrain in ["tokencls", "sptypecls", "spcls"]:
            plt.figure()
            plt.plot(inter_test_all_f1, label="f1")
            plt.legend()
            plt.savefig(result_adress+'/inter_test_plot_f1.png')
        #     plt.show()
            plt.close()
            
    #valid (actucally test)
    if args.dev_exists:
        print('******** Epoch '+str(ep)+' ******** ', file = dev_file)
        dev_result = test(model
                          , pretrain = args.pretrain
                          , baseline = args.baseline
                          , test_or_dev = 'dev'
                          , num_sample = args.dev
                          , train_num = args.stepgame_test_set[0] if args.dataset == "stepgame" else train_num
                          , unseen = False
                          , qtype = args.qtype
                          , other = args.other_var
                          , data_name =   "human" if args.human else args.dataset
                          , save_data = args.save_data
                          , device = device
                          , file = dev_file
                         )
        dev_all_accuracy.append(dev_result[0])
        if args.qtype == 'YN' or args.pretrain in ["tokencls", "sptypecls", "spcls"]:
            f1 = dev_result[1]
            dev_all_f1.append(f1)

        accu = dev_result[0]

        # show image of accuracy  
        plt.figure()
        plt.plot(dev_all_accuracy, label="accuracy")
        plt.legend()
        plt.savefig(result_adress+'/dev_plot_acc.png')
    #     plt.show()
        plt.close()
        if args.qtype == 'YN' or args.pretrain in ["tokencls", "sptypecls", "spcls"]:
            plt.figure()
            plt.plot(dev_all_f1, label="f1")
            plt.legend()
            plt.savefig(result_adress+'/dev_plot_f1.png')
        #     plt.show()
            plt.close()
        
        
    if not args.no_save:
        if args.best_model == 'accuracy' and best_val < accu: 
            torch.save(model, model_address+'/model_'+args.baseline+'_'+args.dataset+'_best_'+args.model+'.th')
            best_val = accu
            no_changes = 0
        elif args.best_model == 'f1' and best_val < f1:
            torch.save(model, model_address+'/model_'+args.baseline+'_'+args.dataset+'_best_'+args.model+'.th')
            best_val = f1
            no_changes = 0
        else: 
            no_changes += 1
    
    # show image of accuracy    
    if args.no_train != True:
        plt.figure()
        plt.plot(all_accuracy, label="accuracy")
        plt.legend()
        plt.savefig(result_adress+'/train_plot_acc.png')
    #     plt.show()
        plt.close()
        if args.qtype == 'YN' or args.pretrain in ["tokencls", "sptypecls", "spcls"]:
            plt.figure()
            plt.plot(all_f1, label="f1")
            plt.legend()
            plt.savefig(result_adress+'/train_plot_f1.png')
        #     plt.show()
            plt.close()

        #show image of losses
        plt.figure()
        plt.plot(all_loss, label="loss")
        plt.legend()
        plt.savefig(result_adress+'/train_plot_loss.png')
    #     plt.show()
        plt.close()


    """
        check if there is three epochs consequently that the result is not better break
        to do this we intialize a variable no_changes which +=1 if there is no changes
    """
    # if no_changes == 10: break

if not args.no_train:
        
    train_file.close()
    inter_test_file.close()
if args.dev_exists:
    dev_file.close()
    
if args.load and args.no_train:
    best_model = model
    best_model.to(device) 
    
elif args.no_train:
    best_model = model
    best_model.to(device) 
    
else:
    best_model = torch.load(model_address+'/model_'+args.baseline+'_'+args.dataset+'_best_'+args.model+'.th', map_location={'cuda:0': 'cuda:'+str(args.cuda),'cuda:1': 'cuda:'+str(args.cuda),'cuda:2': 'cuda:'+str(args.cuda),'cuda:3': 'cuda:'+str(args.cuda), 'cuda:5': 'cuda:'+str(args.cuda), 'cuda:4': 'cuda:'+str(args.cuda), 'cuda:6': 'cuda:'+str(args.cuda),'cuda:7': 'cuda:'+str(args.cuda)})
    best_model.to(device)        

print('~~~~~~~~~~~~ Test ~~~~~~~~~~~~ ')

if not args.human and args.dataset == "stepgame": 
    
    for i in args.stepgame_test_set:
        test_file = open(result_adress+'/test_qa_'+str(i)+'.txt','w')
        test_accuracy = test(best_model
                          , pretrain = args.pretrain
                          , baseline = args.baseline
                          , test_or_dev = 'test'
                          , num_sample = args.test
                          , train_num = i
                          , unseen = False
                          , qtype = args.qtype
                          , other = args.other_var
                        #   , sent_num = i
                          , save_data = args.save_data
                          , device = device
                          , file = test_file
                        ) 
        
        test_file.close()

elif not args.human: 
    test_file = open(result_adress+'/test.txt','w')
    
    test_accuracy = test(best_model
                          , pretrain = args.pretrain
                          , baseline = args.baseline
                          , test_or_dev = 'test'
                          , num_sample = args.test
                          , train_num =  args.stepgame_test_set if args.dataset == "stepgame" else train_num
                          , unseen = False
                          , qtype = args.qtype
                          , other = args.other_var
                          , data_name =  ("human" if args.human else args.dataset)
                          , save_data = args.save_data
                          , device = device
                          , file = test_file
                        ) 
    
    
#     test_all_accuracy.append(test_accuracy) 
    test_file.close()
    
if args.unseentest:
    
    inter_test_unseen_file = open(result_adress+'/unseen_test.txt','w')
    inter_test_unseen_accuracy =  test(best_model
                                  , pretrain = args.pretrain
                                  , baseline = args.baseline
                                  , test_or_dev = 'test'
                                  , num_sample = args.unseen
                                  , train_num = train_num
                                  , unseen = True
                                  , qtype = args.qtype
                                  , other = args.other_var
                                  , data_name =  ("human" if args.human else args.dataset) 
                                  , save_data = args.save_data
                                  , device = device
                                  , file = inter_test_unseen_file
                                ) 
#     inter_test_unseen_all_accuracy.append(inter_test_unseen_accuracy)
    
    inter_test_unseen_file.close()
    
if args.humantest:

    human_file = open(result_adress+'/human_test.txt','w')
    human_accuracy = test(best_model
                      , pretrain = args.pretrain
                      , baseline = args.baseline
                      , test_or_dev = 'test'
                      , num_sample = args.test
                      , train_num = train_num
                      , unseen = False
                      , qtype = args.qtype
                      , other = args.other_var
                      , data_name =  "human"  
                      , save_data = args.save_data
                      , device = device
                      , file = human_file
                    ) 
#     human_all_accuracy.append(human_accuracy)
    
    
    human_file.close()
    
        

#test starts
  
if args.con != 'not' :
    print('~~~~~~~~~~~~ Consistency and Contrast ~~~~~~~~~~~~ ')
    
    
    if args.con == 'consistency':
        con_file = open(result_adress+'/consistency.txt','w') 
        test_accuracy = consistency(model, args.pretrain, args.baseline, args.test, args.qtype, args.other_var, args.human, device, con_file)
        con_file.close()
        
    elif args.con == 'contrast':
        con_file = open(result_adress+'/contrast.txt','w') 
        test_accuracy = contrast(model, args.pretrain, args.baseline, args.test, args.qtype, args.other_var, args.human, device, con_file)
        con_file.close()
    
    elif args.con == 'both':
        cons_file = open(result_adress+'/consistency.txt','w') 
        test_accuracy = consistency(model, args.pretrain, args.baseline, args.test, args.qtype, args.other_var, args.human, device, cons_file)
        cons_file.close()
        
        cont_file = open(result_adress+'/contrast.txt','w') 
        test_accuracy = contrast(model, args.pretrain, args.baseline, args.test, args.qtype, args.other_var, args.human, device, cont_file)
        cont_file.close()
    