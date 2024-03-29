import json
import re
import random
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from Create_LM_input_output import tokenizing, boolean_classification, multiple_classification, initialize_tokenizer
from QA.train import check_answer_equality, concate_input_components

# from BERT import tokenizing
# from ALBERT import tokenizing 
# from XLNet import tokenizing

def train(model
          , criterion
          , optimizer 
          , pretrain = "bertbc"
          , baseline =  "bert"
          , start =  0
          , num_sample =  None
          , train_num =  None
          , qtype =  None
          , data_name = "babi"
          , other =  None
          , device =  "cpu"
          , train_log =  False
          , file =  None
          , epochs = 0
          , batch_size = None
         ):
    
    # initialize_tokenizer(baseline)

    
    candidate_answer = ["Yes", "No"] if qtype == "YN" else ["left", "right", "above", "below"]
    
    model.train()
    
    all_q = 0
    correct = 0
    s_ind = 0
    losses = []
    # k_fold = 7
    if train_num == "train10k":
        training_number = '10k'
    else: 
        training_number = '1k'
        
    address = "dataset/babi/"+training_number+'/'+("17" if qtype == "YN" else "19")+"/train.json"
    
    with open(address) as json_file:
        data = json.load(json_file)

    if qtype == 'YN': TPFN, TP, TPFP = np.array([0]*2), np.array([0]*2), np.array([0]*2)
    
    _temp_batch_input = []
    _temp_batch_answer = []
    
    for s_ind, story in enumerate(tqdm(data['data'][:num_sample])):
        # s_ind+= 1
        # print('sample ',s_ind)
        if s_ind< start:continue
        # samples [epochs*k_fold, (epochs*k_fold)+k_fold] considered as dev
        # if human and s_ind in range((epochs%6)*k_fold, ((epochs%6)*k_fold)+k_fold): continue 

        story_txt = story_txt = story['story'][0]
        
        if train_log: 
            print('sample ',s_ind, file = file)
            print('Story:\n',story_txt, file = file)
        
        #QA tasks
        # each question 
        for question in story['questions']:
            q_text, q_emb= '', []
            
            model.zero_grad()

            all_q += 1
            
            if train_log: print('question: ', question['question'], '\nanswer: ',question['answer'], file = file)  
            
            _temp_batch_input += [concate_input_components([question['question'], story_txt], baseline)]
            _temp_batch_answer += [question['answer']]
            
            if len(_temp_batch_input) < batch_size : continue
                
            if pretrain == 'bertmc':

                loss, output = multiple_classification(model, _temp_batch_input, question['q_type'], candidate_answer, _temp_batch_answer, other = other, device = device, dataset = data_name)


            elif pretrain == 'bertbc':

                loss, output = boolean_classification(model,_temp_batch_input, question['q_type'], candidate_answer, _temp_batch_answer, other = other, device = device, dataset = data_name)

            if train_log: print("predict: ", output, file = file)
            for ind, correct_answer in enumerate(_temp_batch_answer):
                if check_answer_equality(correct_answer, output[ind]):
                    correct+=1
                    if train_log: print('total: ', all_q, ' correct: ', correct, file = file)
                    # print('total: ', all_q, ' correct: ', correct)

                if qtype == 'YN':
                    if correct_answer == ['Yes']: TPFN[0] += 1
                    elif correct_answer == ['No']: TPFN[1] += 1
                    # elif correct_answer == ['DK']: TPFN[2] += 1

                    if output[0] == ['Yes']: TPFP[0] += 1
                    elif output[0] == ['No']: TPFP[1] += 1
                    # elif output == ['DK']: TPFP[2] += 1

                    if output[0] == correct_answer == ['Yes']: TP[0] += 1
                    elif output[0] == correct_answer == ['No']: TP[1] += 1
                    # elif output == correct_answer == ['DK']: TP[2] += 1

        
            if train_log: print("Loss is ", loss.item(), file = file)
            losses += [loss.item()]

            loss.backward()
            optimizer.step()
            
            _temp_batch_answer = []
            _temp_batch_input = []
            



    losses = np.sum(losses)             
    print('Train Final accuracy: ', correct/ all_q)
    print('Train Final accuracy: ', correct/ all_q, file = file)
    print('Losses: ', losses)
    print('Losses: ', losses, file = file)

    if qtype == 'YN':
        print('TP:',TP, '  TPFP: ', TPFP,'   TPFN: ', TPFN ,file = file)
        Precision = np.nan_to_num(TP / TPFP)
        Recall = np.nan_to_num(TP / TPFN)
        F1 = np.nan_to_num((2 * (Precision * Recall)) / (Precision + Recall))
        Macro_F1 = np.average(F1[:2])

        print('Train Final Precision: ', Precision, file = file)

        print('Train Final Recall: ', Recall, file = file)

        print('Train Final F1: ', F1, file = file)

        print('Train Final Macro_F1: ', Macro_F1)
        print('Train Final Macro_F1: ', Macro_F1, file = file)

        return losses, (correct/ all_q, Macro_F1,)

    return losses, (correct/ all_q,)

def correct_token_id(story, question, start_end, tokenizing, file):
    
    story_tokenized = tokenizing(story)
    q_tokenized = tokenizing(question)

    
    #finding the start and end token based on the characters
    sum_char = 0
    start_end_token = []
    for s_e in start_end[:1]:
        temp = s_e[0]
        sum_char = 0
        is_start,start, end = True, None, None
        for ind,word in enumerate(story_tokenized):
            len_word = len(word)
            if temp > sum_char + len(word) : sum_char += len_word; 
            else: 
                if is_start: 
                    start, is_start = ind , False
                    if  s_e[1]-1 <= sum_char + len(word): start_end_token+=[[start, ind]];break 
                    else: temp = s_e[1]-1;
                else: start_end_token+=[[start, ind]]; break
            if ind != len(story_tokenized)-1 and story_tokenized[ind+1] != '.' and story_tokenized[ind+1] != ',' and story_tokenized[ind+1] != "'" and  story_tokenized[ind] != "'": sum_char += 1 # plus one for space
            

        start_end_token[-1][0] += len(q_tokenized)+2 # 2 for [cls] and [SEP]
        start_end_token[-1][1] += len(q_tokenized)+2

    return start_end_token[0]    



def train_babi(model, criterion, optimizer,pretrain, baseline, num_sample, train24k, qtype, other, device, file):
    
    #import baseline
    if baseline == 'bert':
        from BERT import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'xlnet':
        from XLNet import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'albert':
        from ALBERT import question_answering, tokenizing, multiple_choice, boolean_classification
    
    with open('./dataset/babi/train.json') as json_file:
        data = json.load(json_file)
    
    number_samples = int((num_sample/8)+1) if num_sample else num_sample
    
    #random sampling or not
    random.seed(1)
    stories = random.sample(data['data'], number_samples)  if num_sample in [100, 500, 1000, 2000, 5000] else data['data']
    
    
    model.train()
    
    all_q = 0
    correct = 0
    s_ind = 0
    losses = []
    for story in stories[:number_samples]:
        
#         if is_DK_babi(story['story'][0]): continue
            
        s_ind+= 1
        print('sample ',s_ind)
        print('sample ',s_ind, file = file)
        story_txt = story['story'][0]
        
        x = 0
        # each question (span)
        for question in story['questions']:
            q_text, q_emb= '', []
            if question['q_type'] in [qtype] : #and len(question['answer']) == 1: #and x == 0:
                x+=1
                q_text = question['question']
                
                model.zero_grad()
                
                all_q += 1
                print('Story:\n',story_txt, file = file)
                print('question: ', q_text, '\nanswer: ',question['answer'], file = file)  
                
                loss, output = boolean_classification(model, q_text, story_txt, question['q_type'], ['babi'], question['answer'], other, device)
                                                
                #print("logit: ", logit , file = file)
                
                print("predict: ", output, file = file)
                
                correct_answer = question['answer']
                correct_answer.sort()
                if correct_answer == output : 
                    correct+=1
                    print('total: ', all_q, ' correct: ', correct, file = file)
                    print('total: ', all_q, ' correct: ', correct)    

                
                print("Loss is ", loss.item(), file = file)
                losses += [loss.item()]
                
                loss.backward()
                optimizer.step()
                
    losses = np.sum(losses)             
    print('Train Final accuracy: ', correct/ all_q)
    print('Train Final accuracy: ', correct/ all_q, file = file)
    print('Losses: ', losses)
    print('Losses: ', losses, file = file)
    
    return losses, correct/ all_q

def train_boolq(model, criterion, optimizer,pretrain, baseline, num_sample, train24k, qtype, other, device, file):
    
    #import baseline
    if baseline == 'bert':
        from BERT import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'xlnet':
        from XLNet import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'albert':
        from ALBERT import question_answering, tokenizing, multiple_choice, boolean_classification
    
    with open('./dataset/boolQ/train.json') as json_file:
        data = json.load(json_file)
    
    
    model.train()
    
    all_q = 0
    correct = 0
    s_ind = 0
    losses = []
    x = 0
    for story in data['data'][:num_sample]:
        s_ind+= 1
        print('sample ',s_ind)
        print('sample ',s_ind, file = file)
        story_txt = story['passage'][:1000]
#         print(story_txt)
        
        # each question (span)
#         for question in story['questions']:
#             q_text, q_emb= '', []
#             if question['q_type'] in [qtype] : #and len(question['answer']) == 1: #and x == 0:
        x+=1
        q_text = story['question']
        answer = ['Yes'] if story['answer'] == True else ['No']

        model.zero_grad()

        all_q += 1
        print('Story:\n',story_txt, file = file)
        print('question: ', q_text, '\nanswer: ',answer, file = file)  

        
        loss, output = boolean_classification(model, q_text, story_txt, 'YN', ['boolq'], answer, other, device)

        print("predict: ", output, file = file)
        correct_answer = answer
        correct_answer.sort()
        if correct_answer == output : 
            correct+=1
            print('total: ', all_q, ' correct: ', correct, file = file)
            print('total: ', all_q, ' correct: ', correct)    


        print("Loss is ", loss.item(), file = file)
        losses += [loss.item()]

        loss.backward()
        optimizer.step()
                
    losses = np.sum(losses)             
    print('Train Final accuracy: ', correct/ all_q)
    print('Train Final accuracy: ', correct/ all_q, file = file)
    print('Losses: ', losses)
    print('Losses: ', losses, file = file)
    
    return losses, correct/ all_q

def is_DK_babi(story):
    
    has_left = True if 'left' in story else False
    has_right = True if 'right' in story else False
    has_below = True if 'below' in story else False
    has_above = True if 'above' in story else False
    
    if has_left and (has_above or has_below): return True
    elif has_right and (has_above or has_below): return True
    elif has_above and (has_left or has_right): return True
    elif has_below and (has_left or has_right): return True
    
    return False
    
    
    
def question_to_sentence(question, q_type, answer, candidate):
    
    if q_type == 'FB':
        if 'Which' in question:
            question = question.replace('Which block', 'block [MASK]').replace('(s)','').replace('?','.')
        elif 'which' in question:
            question = question.replace('which block', 'block [MASK]').replace('(s)','').replace('?','.')
        elif 'what' in question:    
            question = question.replace('what block', 'block [MASK]').replace('(s)','').replace('?','.')
        else: 
            question = question.replace('What block', 'block [MASK]').replace('(s)','').replace('?','.')
    
    elif q_type == 'FR':
#         print('hi',question)
        if 'What' in question:
            question = question.replace('What is the relation between','').replace('?','.')
            if answer == [4] or answer == [5]: question = question.replace('and', 'is [MASK] [MASK]')
            else: question = question.replace('and', 'is [MASK]')
#             print(question)
        elif 'exist' in question:
            question = question.replace('what relations exist between','').replace('?','.')
            if answer == [4] or answer == [5]: question = question.replace('and', 'is [MASK] [MASK]')
            else: question = question.replace('and', 'is [MASK]')
        elif 'what' in question:
            question = question.replace('what is the relation between','').replace('?','.')
            if answer == [4] or answer == [5]: question = question.replace('and', 'is [MASK] [MASK]')
            else: question = question.replace('and', 'is [MASK]') 
                
        elif 'where' in question:
            question = question.replace('where is','').replace('?','.')
            if answer == [4] or answer == [5]: question = question.replace('regarding to', 'is [MASK] [MASK]')
            else: question = question.replace('regarding to', 'is [MASK]') 
                
        else:
            question = question.replace('Where is','').replace('?','.')
            if answer == [4] or answer == [5]: question = question.replace('regarding to', 'is [MASK] [MASK]')
            else: question = question.replace('regarding to', 'is [MASK]')
    
    elif q_type == 'CO':
#         print(question, answer)
        answer = candidate[answer[0]]
        token_answer = tokenizing(answer)
        mask = ('[MASK] '*len(token_answer))[:-1]
#         print('mask', mask)
        if 'What' in question:
            
            question = question[:question.find('?')+1]
            if 'What object' in question:
                question = question.replace('What object',mask).replace('?','.')
                
            elif 'What thing' in question:
                question = question.replace('What thing',mask).replace('?','.')
            
            elif 'What square' in question:
                question = question.replace('What square',mask).replace('?','.')
                
            else:
                question = question.replace('What',mask).replace('?','.')
               
        elif 'what' in question:
            
            question = question[:question.find('?')+1]
            if 'what object' in question:
                question = question.replace('what object',mask).replace('?','.')
                
            elif 'what thing' in question:
                question = question.replace('what thing',mask).replace('?','.')
                
            else:
                question = question.replace('what',mask).replace('?','.')
            
        elif 'Which' in question:
            if 'Which object' in question:
                question = question[:question.find('?')+1]
                question = question.replace('Which object',mask).replace('?','.')
            elif 'Which square' in question:
                question = question[:question.find('?')+1]
                question = question.replace('Which square',mask).replace('?','.')
            
        elif 'which' in question:
            question = question[:question.find('?')+1]
            question = question.replace('which object',mask).replace('?','.')
       
    return question


def confusion_matrix(truth, predict,correct, TP,TPFP,TPFN):
    
    #Accuracy
#     correct_temp = 0
#     for i in range(len(output)):
#         if output[i] == truth[i].item(): correct_temp+=1
#     correct += correct_temp / len(output)
#     print('total: ', all_q, ' correct: ', correct, file = file)
#     print('total: ', all_q, ' correct: ', correct)
    
#     print(truth, predict)
    if truth == predict : correct +=1
        
    for i in range(len(truth)):
        #TP
        if truth[i] == predict[i]: TP[truth[i]] += 1
        
        #TPFP
        TPFP[predict[i]]+= 1
        
        #TPFN
        TPFN[truth[i]] += 1
            
    
    return correct, TP, TPFP, TPFN

def precision(TP,TPFP):
    return np.nan_to_num(TP[1:]/TPFP[1:])
    
def recall(TP,TPFN):
    return np.nan_to_num(TP[1:]/TPFN[1:])

def F1_measure(TP,TPFP, TPFN,macro= False):

    Precision = np.nan_to_num(TP[1:] / TPFP[1:])
    Recall = np.nan_to_num(TP[1:] / TPFN[1:])
    F1 = np.nan_to_num((2 * (Precision * Recall)) / (Precision + Recall))
    
    return np.average(F1) if macro else F1
