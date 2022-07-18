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


def test(model
         , pretrain =  "bertbc"
         , baseline =  "bert"
         , test_or_dev =  "test"
         , num_sample = None
         , train_num =  '1' #it is the test/dev name
         , unseen = False
         , qtype =  None
         , other =  None  #, sent_num =  '1'
         , save_data =  False
         , data_name = None
         , device =  "cpu"
         , file =  None
         , data = None
         , epochs = 0   
        ):
    
    # initialize_tokenizer(baseline)
    
    all_q = 0
    correct = 0
    correct_no_distance = 0
    # s_ind = 0
    correct_consistency, consistency_total =0, 0
    # k_fold = 7
    
    model.eval()
    
    candidates = ['left', 'right', 'below', 'above', 'lower-left', 'upper-right', 'lower-right', 'upper-left', 'overlap']
    
    # with no auto gradient calculation, torch runs a bit faster
    with torch.no_grad():
        if test_or_dev == "dev":
            if int(train_num) <6: dev_num =  train_num
            else: dev_num = '5'
            with open('/VL/space/rshnk/SpaRT_models/StepGame/Dataset/TrainVersion/qa'+str(dev_num)+'_valid.json') as json_file:
                data = json.load(json_file)
        else:
            test_num = train_num
#             with open('./dataset/new_test.json') as json_file:
            with open('/VL/space/rshnk/SpaRT_models/StepGame/Dataset/TrainVersion/qa'+str(test_num)+'_test.json') as json_file:
                data = json.load(json_file)

        if qtype == 'YN': TPFN, TP, TPFP = np.array([0]*3), np.array([0]*3), np.array([0]*3)

        for s_ind in tqdm(list(data)[:num_sample]):
            # s_ind+= 1
            # print('sample ',s_ind)
            #use k_fold for cross_evaluation
            # if human and test_or_dev == 'dev' and s_ind not in range((epochs%6)*k_fold, ((epochs%6)*k_fold)+k_fold): continue
            
            print('sample ',s_ind, file = file)
            story = data[s_ind]
            story_txt = ' '.join(story['story'])

                # each question (span)
                
            q_text, q_emb= '', []

            q_text = story['question']

            all_q += 1
            print('Story:\n',story_txt, file = file)
            print('question: ', q_text, '\nanswer: ',story['label'], file = file)  
            input_text = concate_input_components([q_text, story_txt], baseline)
            
            
            if pretrain == 'bertmc':

                _, output = multiple_classification(model, [input_text], 'FR', candidates, [], other=other, device = device, dataset = "stepgame")


#             elif pretrain == 'bertbc':

#                 _, output = boolean_classification(model, q_text, story_txt,  'FR', candidates, candidates.index(story['label']), other, device)

            print("predict: ", output[0], file = file)

            correct_answer = [candidates.index(story['label'])]

            if check_answer_equality(correct_answer, output[0]) : 
                correct+=1
                print('total: ', all_q, ' correct: ', correct, file = file)
#                                     print('total: ', all_q, ' correct: ', correct)
                # else: print(s_ind, 'wrong')


 
        print(test_or_dev, ' Final accuracy: ', correct/ all_q)
        print(test_or_dev, ' Final accuracy: ', correct/ all_q, file = file)

        # if qtype == 'YN':

        #     print('TP:',TP, '  TPFP: ', TPFP,'   TPFN: ', TPFN ,file = file)
        #     Precision = np.nan_to_num(TP / TPFP)
        #     Recall = np.nan_to_num(TP / TPFN)
        #     F1 = np.nan_to_num((2 * (Precision * Recall)) / (Precision + Recall))
        #     Macro_F1 = np.average(F1[:2])

        #     print(test_or_dev, ' Final Precision: ', Precision, file = file)

        #     print(test_or_dev, ' Final Recall: ', Recall, file = file)

        #     print(test_or_dev, ' Final F1: ', F1, file = file)

        #     print(test_or_dev, ' Final Macro_F1: ', Macro_F1)
        #     print(test_or_dev, ' Final Macro_F1: ', Macro_F1, file = file)

        #     return (correct/ all_q, Macro_F1)

        return (correct/ all_q,)
    
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
            if temp > sum_char + len(word) : sum_char += len_word
            else: 
                if is_start: 
                    start, is_start = ind , False
                    if  s_e[1]-1 <= sum_char + len(word): start_end_token+=[[start, ind]];break 
                    else: temp = s_e[1]-1
                else: start_end_token+=[[start, ind]]; break
            if ind != len(story_tokenized)-1 and story_tokenized[ind+1] != '.' and story_tokenized[ind+1] != ',' and story_tokenized[ind+1] != "'" and  story_tokenized[ind] != "'": sum_char += 1 # plus one for space

        
        start_end_token[-1][0] += len(q_tokenized)+2 # 2 for [cls] and [SEP]
        start_end_token[-1][1] += len(q_tokenized)+2


    return start_end_token[0]   


def test_babi(model, pretrain, baseline, test_or_dev,num_sample,unseen, qtype, other, device, file):
    
    #import baseline
    if baseline == 'bert':
        from BERT import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'xlnet':
        from XLNet import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'albert':
        from ALBERT import question_answering, tokenizing, multiple_choice, boolean_classification
        
    
    with open('dataset/babi/'+test_or_dev+'.json') as json_file:
        data = json.load(json_file)
    
    all_q = 0
    correct = 0
    s_ind = 0
    
    #random sampling or not
    random.seed(1)
    stories = data['data'] #if other != 'random' else random.sample(data['data'], num_sample)
    
    number_samples = int((num_sample/8)+1) if num_sample else num_sample
    model.eval()
    
    # with no auto gradient calculation, torch runs a bit faster
    with torch.no_grad():
        for story in stories[:number_samples]:
            
#             if is_DK_babi(story['story'][0]): continue
            
            s_ind+= 1
            print('sample ',s_ind)
            print('sample ',s_ind, file = file)
            story_txt = story['story'][0]
            
            
            
            # each question (span)
            for question in story['questions']:
                q_text, q_emb= '', []
                
                if question['q_type'] in [qtype]: 
                    q_text = question['question']
                    
                    
                    
                    all_q += 1
                    print('Story:\n',story_txt, file = file)
                    print('question: ', q_text, '\nanswer: ',question['answer'], file = file)
                    
                    
                    _, output = boolean_classification(model, q_text, story_txt, question['q_type'], ['babi'], question['answer'], other, device)
                    
                    #print("logit: ", logit, file = file)
                    
                    print("predict: ", output, file = file)

                    correct_answer = question['answer']
                    correct_answer.sort()
                    if correct_answer == output : 
                        correct+=1
                        print('total: ', all_q, ' correct: ', correct, file = file)
                        print('total: ', all_q, ' correct: ', correct)
        
    print(test_or_dev,' Final accuracy: ', correct/ all_q)
    print(test_or_dev,' Final accuracy: ', correct/ all_q, file = file)
    
    return correct/ all_q


def test_boolq(model, pretrain, baseline, test_or_dev,num_sample,unseen, qtype, other, device, file):
    
    #import baseline
    if baseline == 'bert':
        from BERT import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'xlnet':
        from XLNet import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'albert':
        from ALBERT import question_answering, tokenizing, multiple_choice, boolean_classification
        
    start_number = 0
    
    if test_or_dev == 'dev':
        with open('dataset/boolQ/dev.json') as json_file:
            data = json.load(json_file)
    else:
        with open('dataset/boolQ/test_1.json') as json_file:
            data = json.load(json_file)
        
        start_number =  len(data['data']) - num_sample
        num_sample = None
    
    all_q = 0
    correct = 0
    s_ind = 0
    
    model.eval()
    
    # with no auto gradient calculation, torch runs a bit faster
    with torch.no_grad():
        for story in data['data'][start_number:num_sample]:
            s_ind+= 1
            print('sample ',s_ind)
            print('sample ',s_ind, file = file)
            story_txt = story['passage'][:1000]


            # each question (span)
#             for question in story['questions']:
#                 q_text, q_emb= '', []
#                 if question['q_type'] in [qtype]: 
            q_text = story['question']+'?'
            answer = ['Yes'] if story['answer'] == True else ['No']
            
            all_q += 1
            print('Story:\n',story_txt, file = file)
            print('question: ', q_text, '\nanswer: ', answer, file = file)  


            _, output = boolean_classification(model, q_text, story_txt, 'YN', ['boolq'], answer, other, device)

            print("predict: ", output, file = file)
            
            correct_answer = answer
            correct_answer.sort()
            if correct_answer == output : 
                correct+=1
                print('total: ', all_q, ' correct: ', correct, file = file)
                print('total: ', all_q, ' correct: ', correct)
        
    print('Test Final accuracy: ', correct/ all_q)
    print('Test Final accuracy: ', correct/ all_q, file = file)
    
    return correct/ all_q



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
    