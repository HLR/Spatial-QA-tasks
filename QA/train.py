import json
import re
import random
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
# from BERT import tokenizing
from Create_LM_input_output import tokenizing, boolean_classification, multiple_classification, initialize_tokenizer
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
          , data_name =  "spartqa"
          , other =  None
          , device =  "cpu"
          , train_log =  False
          , file =  None
          , epochs = 0
          , batch_size = None
         ):
    
    #import baseline
#     if baseline == 'bert':
    # initialize_tokenizer(baseline)
#         from BERT import question_answering, tokenizing, boolean_classification, Masked_LM, Masked_LM_random, token_classification, multiple_classification
#     elif baseline == 'xlnet':
#         from XLNet import question_answering, tokenizing, multiple_choice, boolean_classification
#     elif baseline == 'albert':
#         from ALBERT import question_answering, tokenizing, multiple_choice, boolean_classification

    """
        For computing the batch we store each input (Question+Context+Rule) and seperate thing for amswer. 
        If the size of stored batch is equal to batch_size then we pass them to the model

        For spartuntrain on all (FR and YN) after each passing we change the qtype

        all for spartqa = [CO, FB, FR, and YN]
        amd for spartun = [FR, YN]
    """

    
    model.train()
    
    all_q = 0
    correct = 0
    s_ind = 0
    losses = []
    # k_fold = 7

    task = [qtype]
    qtypes =  ["YN", "FR"]
    # qtypes = ["YN", "FR", "FB", "CO"] if data_name == "spartqa" else ["YN", "FR"]
    qtype = qtypes if qtype == "all" else [qtype]
    

    if data_name == "human":
        with open('./dataset/human_train.json') as json_file:
            data = json.load(json_file)
            data_name = "spartqa"

    elif data_name == "spartun":
        if other == 'simple':
            with open('./dataset/SpaRTUN/train_simple.json') as json_file:
                data = json.load(json_file)
        elif other == "clock":
            with open('./dataset/SpaRTUN/train_clock.json') as json_file:
                data = json.load(json_file)
        else:
            with open('./dataset/SpaRTUN/train.json') as json_file:
                data = json.load(json_file)
        if task == ["all"]: qtype = ["YN"]
        # if qtype == "all": qtype = ["YN"] # change alternatively
        # else: qtype = [qtype]
    else:
        with open('dataset/train.json') as json_file:
            data = json.load(json_file)

    if 'YN' in qtype: TPFN, TP, TPFP = np.array([0]*3), np.array([0]*3), np.array([0]*3)



    _temp_batch_input = []
    _temp_batch_answer = []

    for s_ind, story in enumerate(tqdm(data['data'][:num_sample])):
        # s_ind+= 1
        # print('sample ',s_ind)
        if s_ind< start:continue
        # samples [epochs*k_fold, (epochs*k_fold)+k_fold] considered as dev
        # if human and s_ind in range((epochs%6)*k_fold, ((epochs%6)*k_fold)+k_fold): continue 
        
        story_txt = story['story'][0]
        
        if train_log: 
            print('sample ',s_ind, file = file)
            print('Story:\n',story_txt, file = file)

        x = 0

        
        
        #MLM tasks
        #TODO add batch, add spartun. Now it is set on spartqa
        if pretrain == 'mlm':

            if data_name == "spartqa": tasks_list = ['FB', 'FR', 'CO'] if qtype == 'all' else [qtype]
            else: tasks_list = ['FR', 'YN'] if qtype == 'all' else [qtype]

#             story_txt = 'The circle is above the triangle and the blue square. the blue square is below the circle.'    

            if train_log: print('Story:\n',story_txt, file = file)


            for question in story['questions']:
                q_text, q_emb= '', []
                q_type = question['q_type']
                model.zero_grad()
                if q_type in tasks_list :

                    q_text = question_to_sentence(question['question'], question['q_type'], question['answer'], question['candidate_answers'])

                    if len(question['answer'])> 1 or (q_type == 'FB' and question['answer'] == []) or (q_type == 'FR' and 7 in question['answer']) or (q_type == 'CO' and (question['answer'] == [2] or question['answer'] == [3])): continue

                    answer = question['answer'][0] if q_type == 'FB' else question['candidate_answers'][question['answer'][0]]

                    if q_type == 'CO' and ('which' in answer  or 'in' in answer or 'that' in answer): continue

                    all_q += 1


                    if train_log: print('Question: ',q_text,'\nAnswer: ', answer, file = file)



                    loss, output, truth = Masked_LM(model, story_txt, q_text, answer, other, device, file)

        #             print("predict: ", output)
                    if train_log: print("truth: ", truth, "\npredict: ", output, file = file)
        #             print("truth: ", truth, "\npredict: ", output)

                    correct_temp = 0
                    for i in range(len(output)):
                        if output[i] == truth[i]: correct_temp+=1

                    correct += correct_temp / len(output)
#                         if correct_temp / len(output) == 1:
#                             correct += 1

                    if train_log: print('total: ', all_q, ' correct: ', correct, file = file)
                    print('total: ', all_q, ' correct: ', correct)


                    print("Loss is ", loss.item(), file = file)
                    losses += [loss.item()]

                    loss.backward()
                    optimizer.step() 
        
        #TODO add batch, add spartun. Now it is set on spartqa
        elif pretrain == 'mlmr':
            model.zero_grad()
#             story_txt = 'The circle is above the triangle and the blue square. the blue square is below the circle.'    
            if train_log: print('Story:\n',story_txt, file = file)

            all_q += 1

            loss, output, truth = Masked_LM_random(model, story_txt, s_ind+1, other, device, file)

#             print("predict: ", output)
            if train_log: print("truth: ", truth, "\npredict: ", output, file = file)
#             print("truth: ", truth, "\npredict: ", output)

            correct_temp = 0
            for i in range(len(output)):
                if output[i] == truth[i]: correct_temp+=1

            correct += correct_temp / len(output)
#                         if correct_temp / len(output) == 1:
#                             correct += 1

            if train_log: print('total: ', all_q, ' correct: ', correct, file = file)
            print('total: ', all_q, ' correct: ', correct)


            print("Loss is ", loss.item(), file = file)
            losses += [loss.item()]

            loss.backward()
            optimizer.step()

            
        #QA tasks
        else:
            
            """
                based on the batch_size:

            """
            # print('Story:\n',story_txt, file = file)
            # each question (span)

            for question in story['questions']:
                q_text, q_emb= '', []
                
                model.zero_grad()
                
                
                if question['q_type'] in qtype :
                    if other == 'noDK' and question['answer'] == ['DK']: continue
                    x+=1
                    q_text = question['question']
                    if train_log: print('question: ', q_text, '\nanswer: ',question['answer'], file = file)  
                    all_q += 1

                    """
                        add input and answer to the batch
                        if len(batch) == batch_size pass to the model
                        else: continue
                    """

                    #TODO remove q_text 
                    _temp_batch_input += [concate_input_components([q_text, story_txt], baseline)]
                    _temp_batch_answer += [question['answer']]
                    
                    if len(_temp_batch_input) < batch_size : continue

                    #if batch is full it comes here

                    if pretrain == 'bertmc':

                        loss, output = multiple_classification(model, _temp_batch_input, question['q_type'], question['candidate_answers'], _temp_batch_answer, other = other, device = device, dataset = data_name)

                    elif pretrain == 'bertbc':

                        loss, output = boolean_classification(model, _temp_batch_input, question['q_type'], question['candidate_answers'], _temp_batch_answer, other = other, device = device, dataset = data_name, multi_task = True if task == ["all"] else False)

                    if train_log: print("predict: ", output, file = file)
                    for ind, correct_answer in enumerate(_temp_batch_answer):
                        # correct_answer = question['answer']
                        
                        if check_answer_equality(correct_answer, output[ind]) : 
                            correct+=1
                            if train_log: print('total: ', all_q, ' correct: ', correct, file = file)
                            # print('total: ', all_q, ' correct: ', correct)

                        if question['q_type'] in ['YN']:
                            if correct_answer == ['Yes']: TPFN[0] += 1
                            elif correct_answer == ['No']: TPFN[1] += 1
                            elif correct_answer == ['DK']: TPFN[2] += 1

                            if output[ind] == ['Yes']: TPFP[0] += 1
                            elif output[ind] == ['No']: TPFP[1] += 1
                            elif output[ind] == ['DK']: TPFP[2] += 1

                            if output[ind] == correct_answer == ['Yes']: TP[0] += 1
                            elif output[ind] == correct_answer == ['No']: TP[1] += 1
                            elif output[ind] == correct_answer == ['DK']: TP[2] += 1


                    if train_log: print("Loss is ", loss.item(), file = file)
                    losses += [loss.item()]

                    loss.backward()
                    optimizer.step()

                    _temp_batch_answer = []
                    _temp_batch_input = []
                    if task == ["all"] and data_name == "spartun": 
                        if qtype == ["YN"]: qtype = ["FR"]
                        else: qtype = ["YN"]

    losses = np.sum(losses)             
    print('Train Final accuracy: ', correct/ all_q)
    print('Train Final accuracy: ', correct/ all_q, file = file)
    print('Losses: ', losses)
    print('Losses: ', losses, file = file)

    if "YN" in task or "all" in task :
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

        return losses, (correct/ all_q, Macro_F1)

    return losses, (correct/ all_q,)

def check_answer_equality(correct_answer, prediction):

    correct_answer = [x.lower() if type(x) == str else x for x in correct_answer ]
    correct_answer.sort()
    prediction = [x.lower() if type(x) == str else x for x in prediction ]
    prediction.sort()

    if prediction == correct_answer: return True
    return False

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



def concate_input_components(all_texts, baseline):

    new_input = "" #if baseline == "roberta" else "[CLS] "
    for text in all_texts:
        if new_input != "" and baseline == "roberta": new_input += "<s> "
        new_input += text
        if text != all_texts[-1]: new_input += " </s> " if baseline == "roberta" else " [SEP] "
    
    return new_input

# def train_babi(model, criterion, optimizer,pretrain, baseline, num_sample, train24k, qtype, other, device, file):
    
#     #import baseline
#     if baseline == 'bert':
#         from BERT import question_answering, tokenizing, multiple_choice, boolean_classification
#     elif baseline == 'xlnet':
#         from XLNet import question_answering, tokenizing, multiple_choice, boolean_classification
#     elif baseline == 'albert':
#         from ALBERT import question_answering, tokenizing, multiple_choice, boolean_classification
    
#     with open('./dataset/babi/train.json') as json_file:
#         data = json.load(json_file)
    
#     number_samples = int((num_sample/8)+1) if num_sample else num_sample
    
#     #random sampling or not
#     random.seed(1)
#     stories = random.sample(data['data'], number_samples)  if num_sample in [100, 500, 1000, 2000, 5000] else data['data']
    
    
#     model.train()
    
#     all_q = 0
#     correct = 0
#     s_ind = 0
#     losses = []
#     for story in stories[:number_samples]:
        
# #         if is_DK_babi(story['story'][0]): continue
            
#         s_ind+= 1
#         print('sample ',s_ind)
#         print('sample ',s_ind, file = file)
#         story_txt = story['story'][0]
        
#         x = 0
#         # each question (span)
#         for question in story['questions']:
#             q_text, q_emb= '', []
#             if question['q_type'] in [qtype] : #and len(question['answer']) == 1: #and x == 0:
#                 x+=1
#                 q_text = question['question']
                
#                 model.zero_grad()
                
#                 all_q += 1
#                 print('Story:\n',story_txt, file = file)
#                 print('question: ', q_text, '\nanswer: ',question['answer'], file = file)  
                
#                 loss, output = boolean_classification(model, q_text, story_txt, question['q_type'], ['babi'], question['answer'], other, device)
                                                
#                 #print("logit: ", logit , file = file)
                
#                 print("predict: ", output, file = file)
                
#                 correct_answer = question['answer']
#                 correct_answer.sort()
#                 if correct_answer == output : 
#                     correct+=1
#                     print('total: ', all_q, ' correct: ', correct, file = file)
#                     print('total: ', all_q, ' correct: ', correct)    

                
#                 print("Loss is ", loss.item(), file = file)
#                 losses += [loss.item()]
                
#                 loss.backward()
#                 optimizer.step()
                
#     losses = np.sum(losses)             
#     print('Train Final accuracy: ', correct/ all_q)
#     print('Train Final accuracy: ', correct/ all_q, file = file)
#     print('Losses: ', losses)
#     print('Losses: ', losses, file = file)
    
#     return losses, correct/ all_q

# def train_boolq(model, criterion, optimizer,pretrain, baseline, num_sample, train24k, qtype, other, device, file):
    
#     #import baseline
#     if baseline == 'bert':
#         from BERT import question_answering, tokenizing, multiple_choice, boolean_classification
#     elif baseline == 'xlnet':
#         from XLNet import question_answering, tokenizing, multiple_choice, boolean_classification
#     elif baseline == 'albert':
#         from ALBERT import question_answering, tokenizing, multiple_choice, boolean_classification
    
#     with open('./dataset/boolQ/train.json') as json_file:
#         data = json.load(json_file)
    
    
#     model.train()
    
#     all_q = 0
#     correct = 0
#     s_ind = 0
#     losses = []
#     x = 0
#     for story in data['data'][:num_sample]:
#         s_ind+= 1
#         print('sample ',s_ind)
#         print('sample ',s_ind, file = file)
#         story_txt = story['passage'][:1000]
# #         print(story_txt)
        
#         # each question (span)
# #         for question in story['questions']:
# #             q_text, q_emb= '', []
# #             if question['q_type'] in [qtype] : #and len(question['answer']) == 1: #and x == 0:
#         x+=1
#         q_text = story['question']
#         answer = ['Yes'] if story['answer'] == True else ['No']

#         model.zero_grad()

#         all_q += 1
#         print('Story:\n',story_txt, file = file)
#         print('question: ', q_text, '\nanswer: ',answer, file = file)  

        
#         loss, output = boolean_classification(model, q_text, story_txt, 'YN', ['boolq'], answer, other, device)

#         print("predict: ", output, file = file)
#         correct_answer = answer
#         correct_answer.sort()
#         if correct_answer == output : 
#             correct+=1
#             print('total: ', all_q, ' correct: ', correct, file = file)
#             print('total: ', all_q, ' correct: ', correct)    


#         print("Loss is ", loss.item(), file = file)
#         losses += [loss.item()]

#         loss.backward()
#         optimizer.step()
                
#     losses = np.sum(losses)             
#     print('Train Final accuracy: ', correct/ all_q)
#     print('Train Final accuracy: ', correct/ all_q, file = file)
#     print('Losses: ', losses)
#     print('Losses: ', losses, file = file)
    
#     return losses, correct/ all_q

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
