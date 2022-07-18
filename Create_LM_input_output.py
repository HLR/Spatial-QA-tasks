# checking with BERT

from unittest.util import _MAX_LENGTH
from torchnlp.nn import attention
from transformers import BertTokenizer, BertTokenizerFast, RobertaTokenizer, RobertaTokenizerFast
import torch
import random
import torch.nn as nn


tokenizer, tokenizerFast = None, None

baseline = None

def initialize_tokenizer(lm_model, pretrained_data):
    
    global tokenizer, tokenizerFast, baseline
    
    if lm_model == "bert":
        tokenizer = BertTokenizer.from_pretrained(pretrained_data)
        tokenizerFast = BertTokenizerFast.from_pretrained(pretrained_data)
        baseline = lm_model

    elif lm_model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_data)
        tokenizerFast = RobertaTokenizerFast.from_pretrained(pretrained_data)
        baseline = lm_model
    

def boolean_classification(model, input_text, q_type = None, candidate = [] ,correct_labels = None, other = None, device = "cuda:0", dataset = None, multi_task = False):

    """
        all of the questions with the same task are passed
        first we concat the questions and text and rules 
        and then create the input_ids and token_types
    """
    
    # TODO changed
    encoding = tokenizer(input_text,  return_attention_mask = True, return_tensors="pt", padding=True, max_length = 512)
    # encoding = tokenizer(input_text, max_length=512, pad_to_max_length = True, return_attention_mask = True, return_tensors="pt")

#     if candidate: max_len = max([len(tokenizing(opt)) for opt in candidate])
    
    input_ids, token_type_ids = [], []
    input_ids = encoding["input_ids"].to(device)
    # if baseline == "roberta":
    #     token_type_ids = None
    # else:
    #     token_type_ids = encoding["token_type_ids"].to(device)
    #     for k in range(len(token_type_ids)):
    #         key = list(input_ids[k]).index(102)
    #         for i in range(key):
    #             token_type_ids[k][i] = 1
        
    labels = []
    for correct_label in correct_labels:
        if q_type == 'FR':
            label = torch.tensor([0]*(len(candidate)), device = device).long()
            for opt in correct_label: 
                if dataset in ["spartqa", "babi"]: # answer is the id of correct label
                    label[opt] = 1
                else:
                    label[candidate.index(opt.lower())] = 1

        elif q_type == 'YN' and (other == "noDK" or dataset in[ "sprlqa", "babi", "boolq","spartun"]):

            if correct_label == ['Yes']: label = torch.tensor([1,0], device = device).long()
            elif correct_label == ['No']: label = torch.tensor([0,1], device = device).long()

        elif q_type == 'YN': #spartqa, 

            if correct_label == ['Yes']: label = torch.tensor([1,0,0], device = device).long()
            elif correct_label == ['No']: label = torch.tensor([0,1,0], device = device).long()
            else: label = torch.tensor([0,0,1], device = device).long()
        
        labels += [label]

    if labels == [] : labels = None
    else: labels = torch.stack(labels)
        
    outputs = model(input_ids, labels=labels, task = q_type, multi_task = multi_task, attention_mask = encoding["attention_mask"].to(device))
    # outputs = model(input_ids, labels=labels, task = q_type, multi_task = multi_task, attention_mask = encoding["attention_mask"].to(device))
    
    loss, logits = outputs[:2]
    
    logits = torch.transpose(logits, 0, 1)
    
    outs = []
    
    for logit in logits:
        out = []
        if q_type == 'FR':
            
            #Do some inference
            # remove those relations that cannot happened at the same time and they are both true.
            out_logit = [torch.argmax(log) for log in logit]
            out = Prune_FR_answer_based_on_constraints(out_logit, logit, candidate)
            if dataset not in ["spartqa", "babi"]: out = [candidate[i] for i in out]
        
        
        elif q_type == 'YN' and (len(candidate) == 2 or other == "noDK"):

            max_arg = torch.argmax(logit[:, 1])
    #         print("2", max_arg)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']

        elif q_type == 'YN': #has three candidate labels

            max_arg = torch.argmax(logit[:, 1])
    #         print(correct_label, logits, max_arg)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']
        
        outs += [out]
    
    return loss, outs #, out_logit


    
def multiple_classification(model, input_text, q_type, candidate ,correct_labels , other = None, device = "cpu", dataset = None, multi_task = False):

#     encoding = tokenizer.encode_plus(question, text)
#     print(text, question, candidate, correct_label)
    
#     if candidate: max_len = max([len(tokenizing(opt)) for opt in candidate])
    encoding = tokenizer(input_text, max_length=512, return_tensors="pt", padding=True, return_attention_mask = True)
    # encoding = tokenizer(input_text, max_length=512, pad_to_max_length = True, return_attention_mask = True, return_tensors="pt")
    
    input_ids, token_type_ids = [], []
    input_ids = encoding["input_ids"].to(device)
    if baseline == "roberta":
        token_type_ids = None
    else:
        token_type_ids = encoding["token_type_ids"].to(device)
        for k in range(len(token_type_ids)):

            key = list(input_ids[k]).index(102)
            for i in range(key):
                token_type_ids[k][i] = 1
        
    labels = []
    for correct_label in correct_labels:
        if q_type == 'FR':
            #TODO check this for stepgame
            if dataset not in [ "spartqa", "stepgame"]:
                _correct_label = []
                for label in correct_label:
                    _correct_label += [candidate.index(label.lower())]
            if dataset == "stepgame":
                _correct_label = correct_label[0]
            
            #TODO check this
            else: _correct_label = correct_label[0]
            label = torch.tensor(_correct_label, device = device).long()

        elif q_type == 'YN' and (other == "noDK" or dataset in[ "sprlqa", "babi", "boolq"]):

            if correct_label == ['Yes']: label = torch.tensor(0, device = device).long()
            elif correct_label == ['No']: label = torch.tensor(1, device = device).long()

        elif q_type == 'YN': #spartqa, 

            if correct_label == ['Yes']: label = torch.tensor(0, device = device).long()
            elif correct_label == ['No']: label = torch.tensor(1, device = device).long()
            else: label = torch.tensor(2, device = device).long()
        
        labels += [label]

    if labels == [] : labels = None
    else: labels = torch.stack(labels)

    
    outputs = model(input_ids, labels=labels, token_type_ids = token_type_ids, attention_mask = encoding["attention_mask"].to(device), task = q_type, multi_task = multi_task) #, 
    
    loss, logits = outputs[:2]
    
    outs = []
    
    for logit in logits:
        out = []
        if q_type == 'FR':
            
            max_arg = torch.argmax(logit)
            # max_arg = torch.argmax(logits[0])
            if dataset not in ["stepgame"]:
                out = [candidate[max_arg.item()]]
            else:
                out = [max_arg.item()]

        
        
        elif q_type == 'YN' :

            max_arg = torch.argmax(logit)
            # max_arg = torch.argmax(logits[0])
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK'] # if only two class then this never happen
        
        outs += [out]
            
    
    return loss, outs #, out_logit


def spatial_relation_extraction(model, input_text, labels, device = "cuda:0"):
    
    encoding = tokenizer.batch_encode_plus(input_, max_length=512)
    input_ids, token_type_ids, attention_mask = encoding["input_ids"], encoding["token_type_ids"], encoding['attention_mask']
    
    labels = torch.stack(labels)
    
    outputs = model(input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)
    
    if labels:
        loss, logits = outputs[:2]
    else: 
        loss = None
        logits = outputs[:1]
    
    outs = []
    for logit in logits:
        out_logit = [torch.argmax(log) for log in logit]
        if torch.count_nonzero(out_logit[1:]) > 0:
            out_logit[0] = 0
        else:
            out_logit[0] = 1
        outs += [out_logit]
    
    return loss, outs
        

def Masked_LM(model, text, question, answer, other, device, file):
    
#     print(question, answer)
#     input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    input_ids = tokenizer.encode_plus(question, text, return_tensors="pt")["input_ids"].to(device)

    masked = []
    for ind, i in enumerate(input_ids[0]):
        if i == 103: masked.append(ind)

    token_answer = [tokenizer.convert_tokens_to_ids(i) for i in tokenizing(answer)]
    
    x = [-100] * len(input_ids[0])
    for ind,i in enumerate(masked):
        x[i] = token_answer[ind]
    
    label_ids = torch.tensor([x], device = device)
    
#     print("input_ids2",input_ids)
#     print("label_ids",label_ids)
    
#     segments_ids = torch.tensor([[0]* (len(input_ids))], device = device)
    
    outputs = model(input_ids, labels  = label_ids) #, token_type_ids=segments_ids)
    
#     print(outputs)
    loss, predictions = outputs[:2]
    

    ground_truth = [label_ids[0][i].item() for i in masked]
    truth_token = tokenizer.convert_ids_to_tokens(ground_truth)
    print("truth token: ", truth_token, file = file)
    
    predicted_index = [torch.argmax(predictions[0, predict]).item() for predict in masked ]
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
#     print(predicted_index,predicted_token, file = file)


    print("pred_token: ", predicted_token, file = file)
    
    return loss, predicted_index, ground_truth
    
def Masked_LM_random(model, text, seed_num, other, device, file):
    
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    
#     print("input_ids1",input_ids)
    
    # masked tokens
    masked = []
#     forbiden = [".", ",", "!", "'","?", 'the', 'of']
    random.seed(seed_num)
    masked = random.sample(range(1, len(input_ids[0])-1), int(len(input_ids[0])* 0.12))
#     print("masks: ", masked[0], story_tokenized[masked[0]], file = file)
    
    
    #unchanged tokens, just predict
    not_changed = []
    temp , not_changed_num= 0, int(len(input_ids[0])* 0.015)
    while True:
        x = random.choice(range(1, len(input_ids[0])-1))
        if x not in masked: not_changed.append(x); temp+=1 
        if temp >= not_changed_num: break
    
    #changed tokens
    changed = []
    temp , changed_num= 0, int(len(input_ids[0])* 0.015)
    while True:
        x = random.choice(range(1, len(input_ids[0])-1))
        if x not in masked and x not in not_changed: changed.append(x); temp+=1 
        if temp >= changed_num: break
    
#     print("nums",masked, not_changed, changed)
    
    x = [-100] * len(input_ids[0])
    for i in masked:
        x[i] = input_ids[0][i].item()
        input_ids[0][i] = tokenizer.convert_tokens_to_ids('[MASK]') #103 #[masked]

    for i in not_changed:
        x[i] = input_ids[0][i].item()
    
    for i in changed:
        changed_word =random.choice(range(0,30522))
        x[i] = input_ids[0][i].item()
        input_ids[0][i]= changed_word
    
    label_ids = torch.tensor([x], device = device)
    
#     print("input_ids2",input_ids)
#     print("label_ids",label_ids)
    
    segments_ids = torch.tensor([[0]* (len(input_ids))], device = device)
    
    outputs = model(input_ids, labels  = label_ids, token_type_ids=segments_ids) #, token_type_ids=segments_ids)
    
#     print(outputs)
    loss, predictions = outputs[:2]
    
    predicted = []
    ground_truth = [label_ids[0][i].item() for i in masked] + [label_ids[0][i].item() for i in not_changed] + [label_ids[0][i].item() for i in changed]
    truth_token = tokenizer.convert_ids_to_tokens(ground_truth)
    print("truth token: ", truth_token, file = file)
    
    predicted_index = [torch.argmax(predictions[0, predict]).item() for predict in masked ]
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
#     print(predicted_index,predicted_token, file = file)
    predicted += predicted_index
    
    predicted_index = [torch.argmax(predictions[0, predict]).item() for predict in not_changed ]
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
#     print(predicted_index,predicted_t, file = fileoken)
    predicted += predicted_index
    
    predicted_index = [torch.argmax(predictions[0, predict]).item() for predict in changed ]
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
#     print(predicted_index,predicte, file = filed_token)
    predicted += predicted_index
    
    pred_token = tokenizer.convert_ids_to_tokens(predicted)
    print("pred_token: ", pred_token, file = file)
    
    return loss, predicted, ground_truth

    
# with just triplet classifier
def spatial_classification(model, sentence, triplet, label, device):
#     print('triplet',triplet, label)
    encoding = tokenizer.encode(sentence, add_special_tokens=True)
    token_type_ids = [0]*len(encoding)
    triplets_tokens = []
#     print('encoded sentence', encoding)
    if triplet['trajector'] != [-1,-1] :
        triplets_tokens += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1]+[102] 
        for x in range(triplet['trajector'][0],triplet['trajector'][1]+1):
            token_type_ids[x] = 1 
    else: triplets_tokens += [102]
    
    if triplet['landmark'] != [-1,-1]:
        triplets_tokens += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1]+[102]  
        for x in range(triplet['landmark'][0],triplet['landmark'][1]+1):
            token_type_ids[x] = 1 
    
    else: triplets_tokens += [102]
        
    
    if triplet['spatial_indicator'] != [-1,-1]:
        triplets_tokens += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1]+[102]
        for x in range(triplet['spatial_indicator'][0],triplet['spatial_indicator'][1]+1):
            token_type_ids[x] = 1 
    
#     print('triple tokens: ', triplets_tokens)
    
    token_type_ids = [0]*len(triplets_tokens) + token_type_ids
    encoding =[ encoding[0]]+ triplets_tokens + encoding[1:]
    
    token_type_ids = torch.tensor([token_type_ids], device = device)
    inputs = torch.tensor(encoding, device = device).unsqueeze(0)
    labels = torch.tensor([label], device = device).float()
#     print(inputs.shape, labels.shape)
    
    loss, logits = model(inputs, token_type_ids = token_type_ids, labels=labels)
#     print(outputs)
#     loss = outputs.loss
#     logits = outputs.logits
#     print('logits', logits)
    predicted_index = 1 if logits[0] > 0.5 else 0
#     predicted_index = torch.argmax(logits).item() 
    
#     print('predict', predicted_index)
    return loss, predicted_index#, predicted_index


#triplet classifier and relation tyoe extraction
def spatial_type_classification(model, sentence, triplets, triplet_labels = None, type_labels = None, device= 'cuda:0', asked_class_label = None, other = None, asked = False):


    if other == 'sptypeQA':
        encoding = sentence[0]
        
    else:
        tokenized_text = tokenizing(sentence)

        encoding = tokenizer.encode(sentence, add_special_tokens=True)

    all_token_type_ids, triplets_input = [], []
    for triplet in triplets:

        token_type_ids = [0]*len(encoding)
        triplets_tokens, triplets_tokens_pass, _triplets_tokens_pass = [], [], []
        if triplet['trajector'] not in [['',''], [-1,-1]] :
            if other == 'sptypeQA':
                triplets_tokens += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1].tolist()+[102] 
                triplets_tokens_pass += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1].tolist()#+[1010] 
            elif other == 'triplet':
                triplets_tokens += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1]+[102]
                triplets_tokens_pass += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1]+[102]
            else: 
                triplets_tokens += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1]+[102] 

            for x in range(triplet['trajector'][0],triplet['trajector'][1]+1):
                token_type_ids[x] = 1 
        
        if triplet['spatial_indicator'] not in [['',''], [-1,-1]]:
            
            if other == 'sptypeQA' :
                triplets_tokens += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1].tolist()+[102]
                triplets_tokens_pass += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1].tolist()+[102]#+[1010] 
                
            elif other == 'triplet':
                triplets_tokens += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1]+[102]
                triplets_tokens_pass += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1]+[102]
                
            else:
                triplets_tokens += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1]+[102]
                
                
            for x in range(triplet['spatial_indicator'][0],triplet['spatial_indicator'][1]+1):
                token_type_ids[x] = 1 
    #         print('&&', tokenized_text[triplet['spatial_indicator'][0]-1:triplet['spatial_indicator'][1]]) # shouldn't consider [cls] so subtract 1
            
            spatial_indicator = ''
            if other != 'sptypeQA' and other != 'triplet':
                for z in tokenized_text[triplet['spatial_indicator'][0]-1:triplet['spatial_indicator'][1]]:
                    spatial_indicator += z +' '


        if triplet['landmark'] not in [['',''], [-1,-1]]:
            if other == 'sptypeQA' :
                triplets_tokens += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1].tolist()+[102]  
                triplets_tokens_pass += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1].tolist()#+[1010]  
                
            elif other == 'triplet':
                triplets_tokens += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1]+[102]  
                _triplets_tokens_pass += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1]+[102]  
                
            else:
                triplets_tokens += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1]+[102]  
                            
            for x in range(triplet['landmark'][0],triplet['landmark'][1]+1):
                token_type_ids[x] = 1 

        token_type_ids = [0]*len(triplets_tokens) + token_type_ids + [0]*(26-len(triplets_tokens)) #add PAD to the batch for 6 for larger land and traj and 4 for spatial_ndicator
        all_token_type_ids += [token_type_ids]
        if _triplets_tokens_pass: triplets_tokens_pass = _triplets_tokens_pass   

        if other == 'sptypeQA':
            _encoding =[encoding[0].tolist()]+ triplets_tokens + encoding[1:].tolist()+ [0]*(26-len(triplets_tokens))
            
        else: _encoding = [encoding[0]]+ triplets_tokens + encoding[1:]+ [0]*(26-len(triplets_tokens))
        triplets_input += [_encoding]

    token_type_ids = torch.tensor(all_token_type_ids, device = device)
    inputs = torch.tensor(triplets_input, device = device)
    

    if triplet_labels != None:
        labels = [torch.tensor(triplet_labels, device = device).float(), torch.tensor(type_labels, device = device).long()]
        # labels = [torch.tensor([label], device = device).float(), torch.tensor([type_label], device = device).long(), torch.tensor([asked_class_label if asked_class_label!= None else 0], device = device).float() ]
    #     print(inputs.shape, labels.shape)

        loss, logits1, logits2 = model(inputs, token_type_ids = token_type_ids, labels=labels)
        # loss, logits1, logits2, logits3 = model(inputs, token_type_ids = token_type_ids, labels=labels, asked_compute = asked_class_label)
    #     print(logits1)
    else:
        type_labels = None
        loss = None
        # _,logits1, logits2, logits3 = model(inputs, token_type_ids = token_type_ids, asked_compute = True)
        _,logits1, logits2 = model(inputs, token_type_ids = token_type_ids)
        
    predicted_index1 = [1 if x > 0.65 else 0 for x in logits1[0]]
#     print('logits2', logits2)
    predicted_index2 = [torch.argmax(x).item() for x in logits2[0]]
    # if asked_class_label != None:
    #     predicted_index3 = 1 if logits3[0] > 0.65 else 0
    # else: predicted_index3 = None

#     predicted_index = torch.argmax(logits).item() 

    if triplet_labels == None and (other == 'sptypeQA' or other == ' triplet'):
        if predicted_index1 == 1:
            if other == 'sptypeQA':
                return triplets_tokens_pass
            
            #if we want to pass the triplet in the form of text
            elif other == 'triplet':
                #all_rel_types = ['NaN', 'LEFT','RIGHT','BELOW','ABOVE','NEAR', 'FAR', 'TPP', 'NTPP', 'NTPPI', 'EC']
                triplet_sent = tokenizer.decode(triplets_tokens_pass)
                triplet_sent = triplet_sent[:-5].replace('[SEP]',',')+'.'
                ###For replacing spatial indicator
                #end = triplet_sent.rfind('[SEP]',0,len(triplet_sent)-5)
                #triplet_sent = triplet_sent[:end].replace('[SEP]',',') + '. ' #+all_rel_types[predicted_index2]+'.'
                
#                 print('%%', triplet_sent)
                return triplet_sent
            #else
            
#             triplet_sent = triplet_sent.replace('[SEP]',',')
#             triplet_sent += ' '+all_rel_types[predicted_index2]+'.'
#             triplet_sent = triplet_sent[:-1]+'.'
#             print('In BERT, triplet tokens: ',triplet_sent)
            

        else: return None
            
    elif triplet_labels == None:
        # if logits3:
        return _, predicted_index1, predicted_index2#, predicted_index3
        # else:
        #     return _, predicted_index1, predicted_index2

    else:
        # if logits3:
        return loss, predicted_index1, predicted_index2#, predicted_index3 #, predicted_index
        # else: 
        #     return loss, predicted_index1, predicted_index2 #, predicted_index


def spatial_type_classification_before_batch(model, sentence, triplets, triplet_label = None, type_labels = None, device= 'cuda:0', asked_class_label = None, other = None, asked = False):

#     print(sentence,'\n',triplet)
    #tokenized sentence
    if other == 'sptypeQA':
        encoding = sentence[0]
        
    else:
        tokenized_text = tokenizing(sentence)

        encoding = tokenizer.encode(sentence, add_special_tokens=True)
        
    token_type_ids = [0]*len(encoding)
    
    
#     if other == 'sptypeQA': triplets_tokens, triplets_tokens_pass = [], []
#     else: 
    triplets_tokens, triplets_tokens_pass, _triplets_tokens_pass = [], [], []
#     print('encoded sentence', encoding)
    if triplet['trajector'] != ['','']:
        if other == 'sptypeQA':
            triplets_tokens += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1].tolist()+[102] 
            triplets_tokens_pass += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1].tolist()#+[1010] 
        elif other == 'triplet':
            triplets_tokens += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1]+[102]
            triplets_tokens_pass += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1]+[102]
        else: 
            triplets_tokens += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1]+[102] 

        for x in range(triplet['trajector'][0],triplet['trajector'][1]+1):
            token_type_ids[x] = 1 
    
    if triplet['spatial_indicator'] != ['','']:
        
        if other == 'sptypeQA' :
            triplets_tokens += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1].tolist()+[102]
            triplets_tokens_pass += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1].tolist()+[102]#+[1010] 
            
        elif other == 'triplet':
            triplets_tokens += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1]+[102]
            triplets_tokens_pass += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1]+[102]
            
        else:
            triplets_tokens += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1]+[102]
            
            
        for x in range(triplet['spatial_indicator'][0],triplet['spatial_indicator'][1]+1):
            token_type_ids[x] = 1 
#         print('&&', tokenized_text[triplet['spatial_indicator'][0]-1:triplet['spatial_indicator'][1]]) # shouldn't consider [cls] so subtract 1
        
        spatial_indicator = ''
        if other != 'sptypeQA' and other != 'triplet':
            for z in tokenized_text[triplet['spatial_indicator'][0]-1:triplet['spatial_indicator'][1]]:
                spatial_indicator += z +' '


    if triplet['landmark'] != ['','']:
        if other == 'sptypeQA' :
            triplets_tokens += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1].tolist()+[102]  
            triplets_tokens_pass += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1].tolist()#+[1010]  
            
        elif other == 'triplet':
            triplets_tokens += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1]+[102]  
            _triplets_tokens_pass += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1]+[102]  
            
        else:
            triplets_tokens += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1]+[102]  
                        
        for x in range(triplet['landmark'][0],triplet['landmark'][1]+1):
            token_type_ids[x] = 1 
        
    
    if _triplets_tokens_pass: triplets_tokens_pass += _triplets_tokens_pass   
#     print(spatial_indicator, 'type rel',  type_label, all_rels_type)
#     print()
    token_type_ids = [0]*len(triplets_tokens) + token_type_ids

    if other == 'sptypeQA':
        encoding =[ encoding[0].tolist()]+ triplets_tokens + encoding[1:].tolist()
        
    else: encoding = [encoding[0]]+ triplets_tokens + encoding[1:]
    
    token_type_ids = torch.tensor([token_type_ids], device = device)
    inputs = torch.tensor(encoding, device = device).unsqueeze(0)
    logits3 = None

    if label != None:
        labels = [torch.tensor([label], device = device).float(), torch.tensor([type_label], device = device).long()]
        # labels = [torch.tensor([label], device = device).float(), torch.tensor([type_label], device = device).long(), torch.tensor([asked_class_label if asked_class_label!= None else 0], device = device).float() ]
    #     print(inputs.shape, labels.shape)

        loss, logits1, logits2 = model(inputs, token_type_ids = token_type_ids, labels=labels)
        # loss, logits1, logits2, logits3 = model(inputs, token_type_ids = token_type_ids, labels=labels, asked_compute = asked_class_label)
    #     print(logits1)
    else:
        type_label = None
        loss = None
        # _,logits1, logits2, logits3 = model(inputs, token_type_ids = token_type_ids, asked_compute = True)
        _,logits1, logits2 = model(inputs, token_type_ids = token_type_ids)
        
    predicted_index1 = 1 if logits1[0] > 0.65 else 0
#     print('logits2', logits2)
    predicted_index2 = torch.argmax(logits2[0]).item()
    # if asked_class_label != None:
    #     predicted_index3 = 1 if logits3[0] > 0.65 else 0
    # else: predicted_index3 = None

#     predicted_index = torch.argmax(logits).item() 

    if label == None and (other == 'sptypeQA' or other == ' triplet'):
        if predicted_index1 == 1:
            if other == 'sptypeQA':
                return triplets_tokens_pass
            
            #if we want to pass the triplet in the form of text
            elif other == 'triplet':
                #all_rel_types = ['NaN', 'LEFT','RIGHT','BELOW','ABOVE','NEAR', 'FAR', 'TPP', 'NTPP', 'NTPPI', 'EC']
                triplet_sent = tokenizer.decode(triplets_tokens_pass)
                triplet_sent = triplet_sent[:-5].replace('[SEP]',',')+'.'
                ###For replacing spatial indicator
                #end = triplet_sent.rfind('[SEP]',0,len(triplet_sent)-5)
                #triplet_sent = triplet_sent[:end].replace('[SEP]',',') + '. ' #+all_rel_types[predicted_index2]+'.'
                
#                 print('%%', triplet_sent)
                return triplet_sent
            #else
            
#             triplet_sent = triplet_sent.replace('[SEP]',',')
#             triplet_sent += ' '+all_rel_types[predicted_index2]+'.'
#             triplet_sent = triplet_sent[:-1]+'.'
#             print('In BERT, triplet tokens: ',triplet_sent)
            

        else: return None
            
    elif label == None:
        # if logits3:
        return _, predicted_index1, predicted_index2#, predicted_index3
        # else:
        #     return _, predicted_index1, predicted_index2

    else:
        # if logits3:
        return loss, predicted_index1, predicted_index2#, predicted_index3 #, predicted_index
        # else: 
        #     return loss, predicted_index1, predicted_index2 #, predicted_index
            
    
        


def token_classification(model, text, traj=None, land=None, indicator=None, other=None, device = 'cuda:0', file = ''):    
    
    loss, truth = '', ''
    
    if other != 'sptypeQA':
        inputs = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    else:
        inputs = text
    
    if traj or land or indicator:
        
        labels = make_token_label(text, traj, land, indicator, other).to(device)

        outputs = model(inputs, labels=labels)
        
        loss = outputs.loss
        truth = [element.item() for element in labels[0].flatten()]
    
    else: outputs = model(inputs)
    
    
    logits = outputs.logits
    
    predicted_index = [torch.argmax(predict).item() for predict in logits[0] ]
    
    return loss, predicted_index, truth
    


def make_token_label(text, traj, land, indicator,other):

#     print(text, traj, land, indicator)
    
    encoding = tokenizerFast(text, return_offsets_mapping= True)
    token_starts = [item[0] for item in encoding['offset_mapping']]
    token_ends = [item[1] for item in encoding['offset_mapping']]
#     print(token_starts, token_ends)
#     print()
#     labels_id = ['O', 'B_traj', 'I_traj', 'B_land', 'I_land', 'B_spatial', 'I_spatial']
    labels_id = ['O', 'B_entity', 'I_entity', 'B_indicator', 'I_indicator']
#     text_token = tokenized_text #tokenizing(text)
    labels = torch.tensor([0] * len(encoding['input_ids']))
    
    
    
    #trajector
    for ind, t in enumerate(traj):
        # if t['start']!= '' and t['end']!= '':
#             print(t)
        #skip WRONG ANNOTATION
        if t['start'] not in token_starts and t['end'] not in token_ends: continue
        B_token = token_starts[1:-1].index(t['start'])+1
        E_token = token_ends[1:-1].index(t['end'])+1
#         print(B_token, E_token)
        labels[B_token] = labels_id.index('B_entity')
        for i in range(B_token+1, E_token+1):
            labels[i] = labels_id.index('I_entity')
        
        
    #landmark
    for l in land:
        # if l['start']!= '' and l['end']!= '':
    #         print(l)
        #skip WRONG ANNOTATION
        if l['start'] not in token_starts and l['end'] not in token_ends: continue
        B_token = token_starts[1:-1].index(l['start'])+1
        E_token = token_ends[1:-1].index(l['end'])+1
#         print(B_token, E_token)
        labels[B_token] = labels_id.index('B_entity')
        for i in range(B_token+1, E_token+1):
            labels[i] = labels_id.index('I_entity')
        
    
    #spatial
    for ind in indicator:
        # if ind['start']!= '' and ind['end']!= '':
    #         print(ind)
        #skip WRONG ANNOTATION or it is empty
        if ind['start'] not in token_starts and ind['end'] not in token_ends: continue
        B_token = token_starts[1:-1].index(ind['start'])+1
        E_token = token_ends[1:-1].index(ind['end'])+1
#         print(B_token, E_token)
        labels[B_token] = labels_id.index('B_indicator')
        for i in range(B_token+1, E_token+1):
            labels[i] = labels_id.index('I_indicator')
    
#     print('labels:', labels)

    labels = labels.unsqueeze(0)
    return labels

def extract_entity_token(text, traj, land, indicator, _tuple=False):

#     print(text, traj, land, indicator)
    
    encoding = tokenizerFast(text, return_offsets_mapping= True)
    token_starts = [item[0] for item in encoding['offset_mapping']]
    token_ends = [item[1] for item in encoding['offset_mapping']]   
#     print(token_starts, token_ends)
    if _tuple:
        token_index = {'trajector': [-1,-1], 'landmark': [-1,-1], 'spatial_indicator':[-1,-1], 'rel_type': ''}
#         spatial_indicator = []
        
    else:
        token_index = {'trajector': [-1,-1], 'landmark': [-1,-1], 'spatial_indicator':[-1,-1]}
    #trajector

    if (traj['start']!= '' and traj['start']!= -1) and (traj['end']!= '' and traj['end']!= -1):
#             print(t)
        #start
        token_index['trajector'][0]= token_starts[1:-1].index(traj['start'])+1
        #end
        token_index['trajector'][1]= token_ends[1:-1].index(traj['end'])+1
        
    #landmark
    if (land['start']!= '' and land['start']!= -1) and (land['end']!= '' and land['end']!= -1):
#         print(l)
        #start
        token_index['landmark'][0]= token_starts[1:-1].index(land['start'])+1
        #end
        token_index['landmark'][1]= token_ends[1:-1].index(land['end'])+1
    
    #spatial
    if (indicator['start']!= '' and indicator['start']!= -1) and (indicator['end']!= '' and indicator['end']!= -1):
    
#         if _tuple:
#             spatial_indicator = [token_starts[1:-1].index(indicator['start'])+1, token_ends[1:-1].index(indicator['end'])+1]
# #         print(ind)
#         else:
        token_index['spatial_indicator'][0] = token_starts[1:-1].index(indicator['start'])+1
        token_index['spatial_indicator'][1] = token_ends[1:-1].index(indicator['end'])+1

#     print('token_index:', token_index)
#     if _tuple:
#         return token_index, spatial_indicator
#     else:
    return token_index




# def boolean_classification_end2end(model, question, text, q_type, candidate ,correct_label, other, device):
def boolean_classification_end2end(model, questions, text, q_type, candidates ,correct_labels, other, device, story_annot =None, qs_annot=None, seperate = False):

#     encoding = tokenizer.encode_plus(question, text)
#     print(text, question, candidate, correct_label)
    if seperate:
    #seperate each sentence
        sentences = [h+'.' for h in text.split('. ')]
        sentences[-1] = sentences[-1][:-1]

        text_tokenized = tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"].to(device)
    else:
        text_tokenized = tokenizer(text, return_tensors="pt", padding=True)["input_ids"].to(device)
#     print(text,'\n', text_tokenized)
#     print('&&', questions)
    qs_tokenized = tokenizer(questions, return_tensors="pt", padding=True)["input_ids"].to(device)
#     print(qs_tokenized)
#     print('## text+qs_tokenized: ',text_tokenized)
#     qs_tokenized = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
#     print('### qs_tokenized: ',qs_tokenized)
    
#     inputs = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        
#     if candidate: max_len = max([len(tokenizing(opt)) for opt in candidate])
    
    input_ids, token_type_ids = [], []
    labels =[]
    for _ind,correct_label in enumerate(correct_labels):
        
        if q_type == 'CO':

            label = torch.tensor([[0]]*2, device = device).long()
    #         candid_tokenized = tokenizer(candidate[:2], return_tensors="pt", padding=True)["input_ids"].to(device)
            for opt in candidates[_ind][:2]:
    #             tokenized_opt = tokenizing(opt)
    #             num_tok = len(tokenized_opt)
    #             encoded_options = tokenizer.encode(tokenized_opt + ['[PAD]']*(max_len - num_tok))#[1:]
                input_ids += [encoded_options + encoding["input_ids"][1:]]

            if correct_label == [0] or correct_label == [2]: label[0][0] = 1
            if correct_label == [1] or correct_label == [2]: label[1][0] = 1


        elif q_type == 'FB':

            label = torch.tensor([[0]]*len(candidates[_ind]), device = device).long()
            candid_tokenized = tokenizer(candidates[_ind], return_tensors="pt", padding=True)["input_ids"].to(device)
            for opt in candidates[_ind]:
    #             tokenized_opt = tokenizing(opt)
    # #             num_tok = len(tokenized_opt)
    #             encoded_options = tokenizer.encode(tokenized_opt)#[1:]
                input_ids += [encoded_options + encoding["input_ids"][1:]]

            if 'A' in correct_label: label[0][0] = 1
            if 'B' in correct_label: label[1][0] = 1
            if 'C' in correct_label: label[2][0] = 1

        elif q_type == 'FR':
#             _input = []
            
            label = torch.tensor([0]*7, device = device).long()
            for ind, opt in enumerate(candidates[_ind][:7]): 
#                 _input += [text_tokenized]
                if ind in correct_label:label[ind] = 1
#             text_tokenized = torch.stack(_input)
    #         print('label', label)

    #     elif q_type == 'YN' and other == "DK": #and candidate != ['babi']:
    #         if correct_label == ['Yes']: label = torch.tensor([1,0,0], device = device).long()
    #         elif correct_label == ['No']: label = torch.tensor([0,1,0], device = device).long()
    #         else: label = torch.tensor([0,0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]

    #     elif q_type == 'YN' and other == "noDK":
    #         if correct_label == ['Yes']: label = torch.tensor([1,0], device = device).long()
    #         elif correct_label == ['No']: label = torch.tensor([0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]

    #     elif q_type == 'YN' and candidate == ['boolq']:
    #         if correct_label == ['Yes']: label = torch.tensor([1,0], device = device).long()
    #         elif correct_label == ['No']: label = torch.tensor([0,1], device = device).long()
    # #         else: label = torch.tensor([0,0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]

        elif q_type == 'YN': #and candidate != ['babi']:
            if correct_label == ['Yes']: label = torch.tensor([1,0,0], device = device).long()
            elif correct_label == ['No']: label = torch.tensor([0,1,0], device = device).long()
            else: label = torch.tensor([0,0,1], device = device).long()
#             input_ids = text_tokenized
    #         else : label = torch.tensor([0,0], device = device).long()


    #     elif q_type == 'YN' and candidate == ['babi']:
    #         label = torch.tensor([1,0], device = device).long() if correct_label == ['Yes'] else torch.tensor([0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]    
    
        labels += [label]
    labels = torch.stack(labels).to(device)
#     print('$', correct_labels)
#     print('$$', labels, type(labels))
    
#     input_ids = torch.tensor(input_ids, device = device)
#     print('input shape, label shape',text_tokenized.shape, qs_tokenized.shape, labels.shape)
    if other == 'supervised':
        _outputs = model(text_tokenized, qs_tokenized, story_annotations = story_annot, questions_annotations = qs_annot, labels=labels)
        
    else:
        _outputs = model(text_tokenized, qs_tokenized, labels=labels)
#     print('&&&&&&&&&& outputs', outputs)
    
    losses, outs = [], []
    for outputs in _outputs:
        
        loss, logits = outputs[:2]
        
#         print('$$', loss)
        losses += [loss]
#         print("loss, logits ", loss, logits)
        out_logit = [torch.argmax(log) for log in logits]

        out = [0]
        if q_type == 'FR':

            out = [ind for ind,o in enumerate(out_logit) if o.item() == 1]
            if 2 in out and 3 in out:
                if logits[2][1] >= logits[3][1]:
                    out.remove(3)
                else:
                    out.remove(2)
            if 0 in out and 1 in out:
                if logits[0][1] >= logits[1][1]:
                    out.remove(1)
                else:
                    out.remove(0)
            if 4 in out and 5 in out:
                if logits[4][1] >= logits[5][1]:
                    out.remove(5)
                else:
                    out.remove(4)
            if out == []: out = [7]

        elif q_type == 'FB':

            blocks = ['A', 'B', 'C']
            out = [blocks[ind] for ind,o in enumerate(out_logit) if o.item() == 1]
    #         out = [blocks[ind] for ind,o in enumerate(out_logit) if o.item() == 1]
    #         if 'C' in out and 'C' not in candidate: out.remove('C')

        elif q_type == 'CO':
            out = [ind for ind,o in enumerate(out_logit) if o.item() == 1]
            if 0 in out and 1 in out:
                out = [2]
            elif out == []: out = [3]

        elif q_type == 'YN' and other == 'multiple_class':

            max_arg = torch.argmax(logits)
    #         print(correct_label, logits, max_arg)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']

        elif q_type == 'YN' and other == 'DK' and (candidates == ['babi'] or candidates == ['boolq']):
    #         print('logits: ', logits)
            max_arg = torch.argmax(logits[:2, 1])
    #         print("2", max_arg )
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']

        elif q_type == 'YN' and other == 'DK':

            max_arg = torch.argmax(logits[:, 1])
    #         print("2", max_arg , logits)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']

        elif q_type == 'YN' and other == 'noDK':

            max_arg = torch.argmax(logits[:, 1])
    #         print("2", max_arg)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
    #         else: out = ['DK']

        elif q_type == 'YN' and other == 'change_model':

    #         max_arg = torch.argmax(logits[:, 1])
    #         if max_arg.item() == 0: out = ['Yes']
    #         elif max_arg.item() == 1: out = ['No']
            max_arg = torch.argmax(logits[:, 1])

            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']



        elif q_type == 'YN' and candidates != ['babi']:

            max_arg = torch.argmax(logits[:, 1])

            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']

    #         if out_logit[0] == out_logit[1]:

    #             if out_logit[0].item() == 0: out = ['DK'] 
    #             else: 
    #                 max_arg = torch.argmax(logits[: , 1])
    #                 out = ['Yes'] if max_arg.item() == 0 else ['No']

    #         else: out = ['Yes'] if out_logit[0].item() == 1 else ['No']


        elif q_type == 'YN' and (candidates == ['babi'] or candidates == ['boolq']):

            max_arg = torch.argmax(logits[:, 1])

            out = ['Yes'] if max_arg.item() == 0 else ['No']
        
        outs += [out]
    losses = torch.stack(losses)        
#     print('out logit: ', outs, losses)
    
    return losses, outs #, out_logit

def boolean_classification_addSpRL(model, questions, text, q_type, candidates ,correct_labels, other, device, seperate = False, gt_triplets = None, dataset = None):

#     encoding = tokenizer.encode_plus(question, text)
#     print(text, question, candidate, correct_label)
    z = model.options
    attention_mask_s = None,
    attention_mask_q = None
    if seperate or "q+s" not in model.options :
    #seperate each sentence
        if dataset in ["stepgame", "sprlqa"]:
            sentences = text
        else: 
            sentences = [h+'.' for h in text.split('. ')]
            sentences[-1] = sentences[-1][:-1]
    #sentences = [question] + sentences #the first sentence always is the question
#         print('\nSentences',sentences)
        _text_tokenized = tokenizer(sentences, return_tensors="pt", padding=True, return_attention_mask=True)
        text_tokenized = _text_tokenized["input_ids"].to(device)
        attention_mask_s = _text_tokenized["attention_mask"].to(device)
#         print('sentence', sentences, text_tokenized)
    else:
        text_tokenized = tokenizer(text, return_tensors="pt", padding=True)["input_ids"].to(device)
#     print(text,'\n', text_tokenized)
#     print('Question', questions)
    _qs_tokenized = tokenizer(questions, return_tensors="pt", padding=True, return_attention_mask=True)
    qs_tokenized = _qs_tokenized["input_ids"].to(device)
    attention_mask_q = _qs_tokenized["attention_mask"].to(device)
#     print('## text+qs_tokenized: ',text_tokenized)
#     qs_tokenized = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
#     print('### qs_tokenized: ',qs_tokenized)
    
#     inputs = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        
#     if candidate: max_len = max([len(tokenizing(opt)) for opt in candidate])
    
    input_ids, token_type_ids = [], []
    labels =[]
    for _ind,correct_label in enumerate(correct_labels):
        
        if q_type == 'CO':

            label = torch.tensor([[0]]*2, device = device).long()
    #         candid_tokenized = tokenizer(candidate[:2], return_tensors="pt", padding=True)["input_ids"].to(device)
            for opt in candidates[_ind][:2]:
    #             tokenized_opt = tokenizing(opt)
    #             num_tok = len(tokenized_opt)
    #             encoded_options = tokenizer.encode(tokenized_opt + ['[PAD]']*(max_len - num_tok))#[1:]
                input_ids += [encoded_options + encoding["input_ids"][1:]]

            if correct_label == [0] or correct_label == [2]: label[0][0] = 1
            if correct_label == [1] or correct_label == [2]: label[1][0] = 1


        elif q_type == 'FB':

            label = torch.tensor([[0]]*len(candidates[_ind]), device = device).long()
            candid_tokenized = tokenizer(candidates[_ind], return_tensors="pt", padding=True)["input_ids"].to(device)
    #         for opt in candidates[_ind]:
    # #             tokenized_opt = tokenizing(opt)
    # # #             num_tok = len(tokenized_opt)
    # #             encoded_options = tokenizer.encode(tokenized_opt)#[1:]
    #             input_ids += [encoded_options + encoding["input_ids"][1:]]

            if 'A' in correct_label: label[0][0] = 1
            if 'B' in correct_label: label[1][0] = 1
            if 'C' in correct_label: label[2][0] = 1

        elif q_type == 'FR':

            if dataset != 'stepgame':                
                label = torch.tensor([0]*7, device = device).long()
                for ind, opt in enumerate(candidates[_ind][:7]): 
                    if ind in correct_label:label[ind] = 1

            else:
                label = torch.tensor([correct_label], device=device).long()

    #     elif q_type == 'YN' and other == "DK": #and candidate != ['babi']:
    #         if correct_label == ['Yes']: label = torch.tensor([1,0,0], device = device).long()
    #         elif correct_label == ['No']: label = torch.tensor([0,1,0], device = device).long()
    #         else: label = torch.tensor([0,0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]

        elif q_type == 'YN' and other == "noDK":
            if correct_label == ['Yes']: label = torch.tensor([1,0], device = device).long()
            elif correct_label == ['No']: label = torch.tensor([0,1], device = device).long()
            # input_ids = [encoding["input_ids"]]

    #     elif q_type == 'YN' and candidate == ['boolq']:
    #         if correct_label == ['Yes']: label = torch.tensor([1,0], device = device).long()
    #         elif correct_label == ['No']: label = torch.tensor([0,1], device = device).long()
    # #         else: label = torch.tensor([0,0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]

        elif q_type == 'YN': #and candidate != ['babi']:
            if correct_label == ['Yes']: label = torch.tensor([1,0,0], device = device).long()
            elif correct_label == ['No']: label = torch.tensor([0,1,0], device = device).long()
            else: label = torch.tensor([0,0,1], device = device).long()
#             input_ids = text_tokenized
    #         else : label = torch.tensor([0,0], device = device).long()


    #     elif q_type == 'YN' and candidate == ['babi']:
    #         label = torch.tensor([1,0], device = device).long() if correct_label == ['Yes'] else torch.tensor([0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]    
    
        labels += [label]
    labels = torch.stack(labels).to(device)
#     print('$', correct_labels)
#     print('$$', labels, type(labels))
    
#     input_ids = torch.tensor(input_ids, device = device)
#     print('input shape, label shape',text_tokenized.shape, qs_tokenized.shape, labels.shape)
    
    _outputs, extracted_triplets_index = model(text_tokenized, qs_tokenized, labels=labels, attention_mask_s = attention_mask_s, attention_mask_q = attention_mask_q, gt_triplets = gt_triplets)
#     print('&&&&&&&&&& outputs', outputs)
    
    losses, outs = [], []
    for outputs in _outputs:
        
        loss, logits = outputs[:2]
        
#         print('$$', loss)
        losses += [loss]
#         print("loss, logits ", loss, logits)
        out_logit = [torch.argmax(log) for log in logits]

        out = [0]
        if q_type == 'FR':
            if dataset != 'stepgame':
                out = [ind for ind,o in enumerate(out_logit) if o.item() == 1]
                if 2 in out and 3 in out:
                    if logits[2][1] >= logits[3][1]:
                        out.remove(3)
                    else:
                        out.remove(2)
                if 0 in out and 1 in out:
                    if logits[0][1] >= logits[1][1]:
                        out.remove(1)
                    else:
                        out.remove(0)
                if 4 in out and 5 in out:
                    if logits[4][1] >= logits[5][1]:
                        out.remove(5)
                    else:
                        out.remove(4)
                if out == []: out = [7]
            else:
                out = [out_logit[0].item()]

        elif q_type == 'FB':

            blocks = ['A', 'B', 'C']
            out = [blocks[ind] for ind,o in enumerate(out_logit) if o.item() == 1]
    #         out = [blocks[ind] for ind,o in enumerate(out_logit) if o.item() == 1]
    #         if 'C' in out and 'C' not in candidate: out.remove('C')

        elif q_type == 'CO':
            out = [ind for ind,o in enumerate(out_logit) if o.item() == 1]
            if 0 in out and 1 in out:
                out = [2]
            elif out == []: out = [3]

        elif q_type == 'YN' and other == 'multiple_class':

            max_arg = torch.argmax(logits)
    #         print(correct_label, logits, max_arg)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']

        elif q_type == 'YN' and other == 'DK' and (candidates == ['babi'] or candidates == ['boolq']):
    #         print('logits: ', logits)
            max_arg = torch.argmax(logits[:2, 1])
    #         print("2", max_arg )
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']

        elif q_type == 'YN' and other == 'DK':

            max_arg = torch.argmax(logits[:, 1])
    #         print("2", max_arg , logits)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']

        elif q_type == 'YN' and other == 'noDK':

            max_arg = torch.argmax(logits[:, 1])
    #         print("2", max_arg)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
    #         else: out = ['DK']

        elif q_type == 'YN' and other == 'change_model':

    #         max_arg = torch.argmax(logits[:, 1])
    #         if max_arg.item() == 0: out = ['Yes']
    #         elif max_arg.item() == 1: out = ['No']
            max_arg = torch.argmax(logits[:, 1])

            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']



        elif q_type == 'YN' and candidates != ['babi']:

            max_arg = torch.argmax(logits[:, 1])

            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']

    #         if out_logit[0] == out_logit[1]:

    #             if out_logit[0].item() == 0: out = ['DK'] 
    #             else: 
    #                 max_arg = torch.argmax(logits[: , 1])
    #                 out = ['Yes'] if max_arg.item() == 0 else ['No']

    #         else: out = ['Yes'] if out_logit[0].item() == 1 else ['No']


        elif q_type == 'YN' and (candidates == ['babi'] or candidates == ['boolq']):

            max_arg = torch.argmax(logits[:, 1])

            out = ['Yes'] if max_arg.item() == 0 else ['No']
        
        outs += [out]
    losses = torch.stack(losses)        
#     print('out logit: ', outs, losses)
    
    return losses, outs, extracted_triplets_index #, out_logit


def tokenizing(text):
    
    encoding = tokenizer.tokenize(text)
    
    return encoding


def Prune_FR_answer_based_on_constraints(out_logit,  logits, candidate_answer):
    
    

    """
        for i in answer:
            check all the incompatibel rels. add those which are 1, from those find the highest score. add that to the final answer.
            
        if "DK" in candiate answer consider empty else choose the highest true
    """
    
    #translate the candidate answers in spartqa to relation_types
    if candidate_answer == ['left', 'right', 'above', 'below', 'near to', 'far from', 'touching', 'DK']:
        candidate_answer = ['left', 'right', 'above', 'below', 'near', 'far', 'ec', 'DK'] #touching is alway EC since we only ask about relations between blocks in spartqa


    out = [ind for ind,o in enumerate(out_logit) if o.item() == 1]
        
    selected_output = []
    checked_rel = []
    for ans in out:
        if ans in checked_rel: continue
        checked_rel += [ans]
        all_incomp_index = [candidate_answer.index(incomp_rel.lower()) for incomp_rel in incompatible_relations[candidate_answer[ans].upper()] if incomp_rel.lower() in candidate_answer and candidate_answer.index(incomp_rel.lower()) in out]
        
        if all_incomp_index == [] : 
            selected_output += [ans]
            continue
        #select the highest
        max_id = ans
        # checked_rel = []
        for ind in all_incomp_index:
            checked_rel += [ind]
            if logits[ind][1] >= logits[max_id][1]:
                max_id = ind
        selected_output += [max_id]
    
    if selected_output: out = selected_output
    
    if out == []: 
        if "DK" in candidate_answer:
            out = [candidate_answer.index("DK")]
        else:
            out = [torch.argmax(logits[:, 1]).item()]

    return out
    
                
    
incompatible_relations = {"DC": ["EC", "PO", "EQ", "NTPP", "NTPPI", "TPP", "TPPI"], 
           "EC": ["DC", "PO", "EQ", "NTPP", "NTPPI", "TPP", "TPPI"], 
           "PO": ["DC", "EC", "EQ", "NTPP", "NTPPI", "TPP", "TPPI"], 
           "NTPP": ["DC", "EC", "PO", "EQ", "NTPPI", "TPP", "TPPI"], 
           "NTPPI": ["DC", "EC", "PO", "EQ", "NTPP", "TPP", "TPPI"], 
           "TPP": ["DC", "EC", "PO", "EQ", "NTPP", "NTPPI", "TPPI"], 
           "TPPI": ["DC", "EC", "PO", "EQ", "NTPP", "NTPPI", "TPP"],
           "EQ": ["DC", "EC", "PO", "NTPP", "NTPPI", "TPP", "TPPI"],
           'RIGHT': ["LEFT"], 
           'LEFT':["RIGHT"], 
           'BELOW':["ABOVE"], 
           'ABOVE':["BELOW"],
           "BEHIND": ["FRONT"],
           "FRONT": ["BEHIND"],
           'FAR':["NEAR"], 
           'NEAR':["FAR"]
          }
            
#     reverse = {"DC": ["EC", "PO", "EQ", "NTPP", "NTPPI", "TPP", "TPPI"], 
#                "EC": ["DC", "PO", "EQ", "NTPP", "NTPPI", "TPP", "TPPI"], 
#                "PO": ["DC", "EC", "EQ", "NTPP", "NTPPI", "TPP", "TPPI"], 
#                "NTPP": ["DC", "EC", "PO", "EQ", "NTPPI", "TPP", "TPPI"], 
#                "NTPPI": ["DC", "EC", "PO", "EQ", "NTPP", "TPP", "TPPI"], 
#                "TPP": ["DC", "EC", "PO", "EQ", "NTPP", "NTPPI", "TPPI"], 
#                "TPPI": ["DC", "EC", "PO", "EQ", "NTPP", "NTPPI", "TPP"],
#                "EQ": ["DC", "EC", "PO", "NTPP", "NTPPI", "TPP", "TPPI"],
#                'RIGHT': ["LEFT"], 
#                'LEFT':["RIGHT"], 
#                'BELOW':["ABOVE"], 
#                'ABOVE':["BELOW"],
#                "BEHIND": ["FRONT"],
#                "FRONT": ["BEHIND"],
#                'FAR':["FAR"], 
#                'NEAR':["NEAR"]
#               }
    

#     return reverse[rel]




