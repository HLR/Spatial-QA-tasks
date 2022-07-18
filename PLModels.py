# from transformers import BertPreTrainedModel, BertModel, BertOnlyMLMHead
from transformers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from typing import Union, List
import numpy as np

from torch.autograd import Variable

def weights_init_normal(m):
    """Takes in a module and initializes all linear layers with weight
    values taken from a normal distribution."""

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find("Linear") != -1:
        try:
            y = m.in_features
            # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0, 1 / np.sqrt(y))
            # m.bias.data should be 0
            m.bias.data.fill_(0)
        except:
            # bound = 1 / math.sqrt(m.weight.size(-1))
            # torch.nn.init.uniform_(m.weight.data, -bound, bound)
            y = m.in_channels
            # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0, 1 / np.sqrt(y))
            print(m)
            # raise

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

    
class BertBooleanQuestionYN(BertPreTrainedModel):
    
    def __init__(self, config, 
                device = 'cuda:0', 
                no_dropout = False, 
                num_labels_YN = 2, 
                dataset = "spartun"# LM = "bert",# has_batch = False, # criterion = 'cross'
                ):
        
        super().__init__(config)
 #LM = "bert", #has_batch = False, #criterion = "cross"

        if no_dropout:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        
        self.device2 = device
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes = 2
        self.num_labels = num_labels_YN

        if dataset == "spartqa":
            self.alphas = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.27, 0.73] ]).to(self.device2)
        elif dataset == "human":
            self.alphas = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.1, 0.9] ]).to(self.device2)
        elif dataset == "spartun":
            self.alphas = torch.tensor([[0.55, 0.45], [0.45, 0.55]]).to(self.device2)
        else:
            self.alphas = torch.tensor([[0.5, 0.5]]*self.num_labels_YN).to(self.device2)

        # self.classifier = nn.Linear(config.hidden_size, self.num_classes)

        
        classifiers = []
        self.criterion = []
        for item in range(self.num_labels):
            classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
            self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
        self.classifiers = nn.ModuleList(classifiers)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        

    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task = None,
        multi_task = None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        
        pooled_output = self.dropout(pooled_output)

        logits = []
        
        for ind in range(self.num_labels): 
            logit = self.classifiers[ind](pooled_output)
            logits.append(logit)
            # logits.append(logit.squeeze(0))

        if labels is not None:

            loss = 0
            out_logits = []
            for ind, logit in enumerate(logits):
                loss += self.criterion[ind](logit, labels[:, ind])
                # loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
                out_logits.append(self.softmax(logit))
            outputs = (loss, torch.stack(out_logits),) + outputs[2:]
            
        else:
            out_logits = []
            for ind, logit in enumerate(logits):
                out_logits.append(self.softmax(logit))
            outputs = (None, torch.stack(out_logits),) + outputs[2:]

        return outputs 
    
class BertBooleanQuestionFR(BertPreTrainedModel):
    

        
    def __init__(self, config, 
                device = 'cuda:0', 
                no_dropout = False, 
                num_labels_FR = 7, 
                dataset = "spartun"# LM = "bert",# has_batch = False, # criterion = 'cross'
                ):
        
        super().__init__(config)
        
        if no_dropout:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
            
        self.device1 = device
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes = 2
        self.num_labels = num_labels_FR
        
        # self.classifier = nn.Linear(config.hidden_size, self.num_classes)

        if dataset == "spartqa":
                self.alphas = torch.tensor([[0.20, 0.8], [0.20, 0.8], [0.25, 0.75], [0.4, 0.6], [0.1, 0.9], [0.2, 0.98], [0.2, 0.98]]).to(self.device1)
            # self.alphas = torch.tensor([[0.15, 0.85], [0.25, 0.75], [0.25, 0.75], [0.4, 0.6], [0.06, 0.94], [0.01, 0.99] ,[0.01, 0.99]]).to(self.device1)
            # self.alphas = torch.tensor([[0.20, 0.8], [0.20, 0.8], [0.25, 0.75], [0.4, 0.6], [0.05, 0.95], [0.01, 0.99] ,[0.01, 0.99]]).to(self.device1)
            # self.alphas = torch.tensor([[0.20, 0.8], [0.20, 0.8], [0.25, 0.75], [0.4, 0.6], [0.05, 0.95], [0.01, 0.99] ,[0.01, 0.99], [0.01, 0.99]]).to(self.device1)
        elif dataset == "human":
            self.alphas = torch.tensor([[0.35, 0.65], [0.25, 0.75], [0.25, 0.75], [0.2, 0.8], [0.25, 0.75], [0.45, 0.55], [0.05, 0.95]]).to(self.device1)
        elif dataset == "spartun":
            self.alphas = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.25, 0.75], [0.25, 0.75], [0.1, 0.9], [0.1, 0.9], [0.04, 0.96], [0.15, 0.85], [0.24, 0.76], [0.07, 0.93], [0.02, 0.98], [0.05, 0.95], [0.12, 0.88], [0.05, 0.95], [0.1, 0.9]]).to(self.device1)
        elif dataset == "babi":
            self.alphas = torch.tensor([[0.60, 0.4], [0.57, 0.43], [0.6, 0.4], [0.41, 0.59]]).to(self.device1)
        else: 
            self.alphas = torch.tensor([[0.5, 0.5]]*self.num_labels).to(self.device1)
            
        classifiers = []
        self.criterion = []
        for item in range(self.num_labels):
            classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
            self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
        self.classifiers = nn.ModuleList(classifiers)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task = None,
        multi_task = None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        pooled_output = outputs[1]

        
        pooled_output = self.dropout(pooled_output)
#         print(pooled_output.shape)
        logits = []
    
        for ind  in range(self.num_labels): 
            logit = self.classifiers[ind](pooled_output)
            # logit = self.classifiers[ind](pooled_output[ind])
            logits.append(logit)
        
        # for check on YN
#         for ind in range(7): 
#             logit = self.classifiers[ind](pooled_output)
#             logits.append(logit)
#         print("FR",logits)

        if labels is not None:

            loss = 0
            out_logits = []
            for ind, logit in enumerate(logits):

                loss += self.criterion[ind](logit, labels[:, ind])
                # loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
                out_logits.append(self.softmax(logit))
            outputs = (loss, torch.stack(out_logits),) + outputs[2:]
        else:
            out_logits = []
            for ind, logit in enumerate(logits):
                out_logits.append(self.softmax(logit))
            outputs = (None, torch.stack(out_logits),) + outputs[2:]

        return outputs  # (loss), reshaped_logits, (hidden_st


class BertMultipleClass(BertPreTrainedModel):
    
    def __init__(self, config, 
                device = 'cuda:0', 
                no_dropout = False, 
                num_classes = 3,
                dataset = "spartun",
                qtype = "FR"
                ):
        super().__init__(config)

        if no_dropout:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        
        self.device1 = device
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes = num_classes
        self.qtype = qtype

        # self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        
        if self.qtype == 'YN':
            if dataset == "spartqa":
                self.alphas = torch.tensor([0.67, 1.3, 1.32]).to(self.device1)
            else:
                self.alphas = torch.tensor([1/self.num_classes]*self.num_classes).to(self.device1)
        elif self.qtype == "FR":
            # self.alphas = torch.tensor([0.125]*self.num_classes).to(self.device1)
            self.alphas = torch.tensor([1/self.num_classes]*self.num_classes).to(self.device1)

        # classifiers = []
        # self.criterion = []
        # for item in range(1):
        self.classifiers= nn.Linear(config.hidden_size, self.num_classes)
        self.criterion= FocalLoss(alpha=self.alphas, class_num=self.num_classes, gamma = 2)
        # self.classifiers = nn.ModuleList(classifiers)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        

    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        
        pooled_output = self.dropout(pooled_output)

        # for ind in range(1): 
        logit = self.classifiers(pooled_output)

        # logits = logit.squeeze(0)
        out_logits = self.softmax(logit)
        
        if labels is not None:

            # loss = 0
            # out_logits = []
            #check labels
            loss = self.criterion(logit, labels)
            
            outputs = (loss, out_logits,) + outputs[2:]
        else:
            outputs = (None, out_logits,) + outputs[2:]
            

        return outputs



class BertMultiTaskMultipleClass(BertPreTrainedModel):
    #TODO add criterion to load
    def __init__(self, config, 
                device = 'cuda:0', 
                no_dropout = False, 
                num_classes_YN = None, 
                num_classes_FR = None, 
                dataset = "stepgame", 
                LM = "bert",
                has_batch = False,
                criterion = 'focal'
                ):
        super().__init__(config)

        if no_dropout:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        
        self.device1 = device

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes_YN = num_classes_YN
        self.num_classes_FR = num_classes_FR
#         self.qtype = qtype

        if self.num_classes_YN:
            if dataset == "spartqa":
                self.alphasYN = torch.tensor([0.67, 1.3, 1.32]).to(self.device1)
            elif dataset == "spartun":
                self.alphasYN = torch.tensor([0.4, 0.6]).to(self.device1)
            else:
                self.alphasYN = torch.tensor([1/self.num_classes_YN]*self.num_classes_YN).to(self.device1)
            #TODO test MLP, add batch
            self.classifiers_YN = nn.Linear(config.hidden_size, self.num_classes_YN)                                           
            if criterion == "cross":
                self.criterion_YN = nn.CrossEntropyLoss(weight=self.alphas)
            else:
                self.criterion_YN = FocalLoss(alpha=self.alphasYN, class_num=self.num_classes_YN, gamma = 2)

#             self.init_weights()
#             self.classifiers_YN.apply(weights_init_normal)

        if self.num_classes_FR:
            # if dataset == "spartqa":
            # else:
            #TODO compute for stepgame
            self.alphasFR = torch.tensor([1/self.num_classes_FR]*self.num_classes_FR).to(self.device1)
            
            self.classifiers_FR =  nn.Linear(config.hidden_size, self.num_classes_FR)
            if criterion == "cross":                                 
                self.criterion_FR = nn.CrossEntropyLoss(weight=self.alphas)
            else:
                self.criterion_FR =FocalLoss(alpha=self.alphasFR, class_num=self.num_classes_FR, gamma = 2, size_average=False)
        
        
            # self.classifiers_FR.apply(weights_init_normal)


        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task = "YN",
        multi_task = False
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if task == "YN":
            logit = self.classifiers_YN(pooled_output)
        elif task == "FR":
            logit = self.classifiers_FR(pooled_output)

        out_logits = self.softmax(logit)
        if labels is not None:

            if task == "YN":
                loss = self.criterion_YN(logit, labels)
            elif task == "FR":
                loss = self.criterion_FR(logit, labels)
            outputs = (loss, out_logits,) + outputs[2:]
        
        else:
            outputs = (None,out_logits,)+ outputs[2:]

        return outputs

class BertMultiTaskMultipleClassLoad(BertPreTrainedModel):
    #TODO add criterion to load
    def __init__(self, config, 
                device = 'cuda:0', 
                no_dropout = False, 
                num_classes_YN = None, 
                num_classes_FR = None, 
                dataset = "stepgame", 
                LM = "bert",
                has_batch = False,
                criterion = 'focal'
                ):
        super().__init__(config)

        if no_dropout:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        
        self.device1 = device

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes_YN = num_classes_YN
        self.num_classes_FR = num_classes_FR
#         self.qtype = qtype

        if self.num_classes_YN:
            if dataset == "spartqa":
                self.alphasYN = torch.tensor([0.67, 1.3, 1.32]).to(self.device1)
            elif dataset == "spartun":
                self.alphasYN = torch.tensor([0.4, 0.6]).to(self.device1)
            else:
                self.alphasYN = torch.tensor([1/self.num_classes_YN]*self.num_classes_YN).to(self.device1)
            #TODO test MLP, add batch
            self.classifiers_YN_load = nn.Linear(config.hidden_size, self.num_classes_YN)

                                            
            if criterion == "cross":
                self.criterion_YN = nn.CrossEntropyLoss(weight=self.alphas)
            else:
                self.criterion_YN = FocalLoss(alpha=self.alphasYN, class_num=self.num_classes_YN, gamma = 2)

#             self.init_weights()
#             self.classifiers_YN.apply(weights_init_normal)

        if self.num_classes_FR:
            # if dataset == "spartqa":
            # else:
            #TODO compute for stepgame
            self.alphasFR = torch.tensor([1/self.num_classes_FR]*self.num_classes_FR).to(self.device1)
            
            self.classifiers_FR_load =  nn.Linear(config.hidden_size, self.num_classes_FR)
            if criterion == "cross":                                 
                self.criterion_FR = nn.CrossEntropyLoss(weight=self.alphas)
            else:
                self.criterion_FR =FocalLoss(alpha=self.alphasFR, class_num=self.num_classes_FR, gamma = 2, size_average=False)
        
        
            # self.classifiers_FR.apply(weights_init_normal)


        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task = "YN",
        multi_task = False
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if task == "YN":
            logit = self.classifiers_YN_load(pooled_output)
        elif task == "FR":
            logit = self.classifiers_FR_load(pooled_output)

        out_logits = self.softmax(logit)
        if labels is not None:

            if task == "YN":
                loss = self.criterion_YN(logit, labels)
            elif task == "FR":
                loss = self.criterion_FR(logit, labels)
            outputs = (loss, out_logits,) + outputs[2:]
        
        else:
            outputs = (None,out_logits,)+ outputs[2:]

        return outputs

#TODO bertpretraining is added
class BertMultiTaskBooleanQuestion(BertPreTrainedModel):
    
    def __init__(self, config, 
                device = 'cuda:0', 
                no_dropout = False, 
                num_labels_YN = None, 
                num_labels_FR = None,  
                dataset = "spartqa",
                LM = "bert",
                has_batch = False,
                criterion = "focal"
                ):
        super().__init__(config)
        
        if no_dropout:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
            
        self.device1 = device

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes = 2
        self.num_labels_YN = num_labels_YN
        self.num_labels_FR = num_labels_FR
        # self.qtype = qtype

        if self.num_labels_YN:
            if dataset == "spartqa":
                self.alphasYN = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.27, 0.73] ]).to(self.device1)
            elif dataset == "human":
                self.alphasYN = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.1, 0.9] ]).to(self.device1)
            elif dataset == "spartun":
                self.alphasYN = torch.tensor([[0.55, 0.45], [0.45, 0.55]]).to(self.device1)
            else:
                self.alphasYN = torch.tensor([[0.5, 0.5]]*self.num_labels_YN).to(self.device1)
            
            #initialize the classifier model
            classifiers_YN = []
            self.criterion_YN = []
            for item in range(self.num_labels_YN):
                classifiers_YN.append(nn.Linear(config.hidden_size, self.num_classes))
                
                if criterion == "cross":
                    self.criterion_YN.append(nn.CrossEntropyLoss(weight=self.alphasYN[item]))
                else:
                    self.criterion_YN.append(FocalLoss(alpha=self.alphasYN[item], class_num=self.num_classes, gamma = 2))
            self.classifiers_YN = nn.ModuleList(classifiers_YN)
#             self.sigma_YN = nn.Parameter(torch.ones(1))


        if self.num_labels_FR:
            if dataset == "spartqa":
                self.alphasFR = torch.tensor([[0.20, 0.8], [0.20, 0.8], [0.25, 0.75], [0.4, 0.6], [0.1, 0.9], [0.02, 0.98], [0.02, 0.98]]).to(self.device1)
            elif dataset == "human":
                self.alphasFR = torch.tensor([[0.35, 0.65], [0.25, 0.75], [0.25, 0.75], [0.2, 0.8], [0.25, 0.75], [0.45, 0.55], [0.05, 0.95]]).to(self.device1)
            elif dataset == "spartun":
                self.alphasFR = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.27, 0.73], [0.26, 0.74], [0.1, 0.9], [0.1, 0.9], [0.02, 0.98], [0.15, 0.85], [0.26, 0.74], [0.07, 0.93], [0.002, 0.998], [0.02, 0.98], [0.1, 0.9], [0.03, 0.97], [0.08, 0.92]]).to(self.device1)
            elif dataset == "babi":
                self.alphasFR = torch.tensor([[0.60, 0.4], [0.57, 0.43], [0.6, 0.4], [0.41, 0.59]]).to(self.device1)
            else:
                self.alphasFR = torch.tensor([[0.5, 0.5]]*self.num_labels_FR)



            classifiers_FR = []
            self.criterion_FR = []
            #TODO add batch
            #TODO changed

            for item in range(self.num_labels_FR):
                classifiers_FR.append(nn.Linear(config.hidden_size, self.num_classes))
                if criterion == "corss":
                    self.criterion_FR.append(nn.CrossEntropyLoss(weight=self.alphasFR[item]))
                else:
                    self.criterion_FR.append(FocalLoss(alpha=self.alphasFR[item], class_num=self.num_classes, gamma = 2, size_average=True))

            self.classifiers_FR = nn.ModuleList(classifiers_FR)
                # self.sigma_FR = nn.Parameter(torch.ones(1))
        
        # self.sigma = nn.Parameter(torch.ones(2))
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        
        # self.classifiers_FR.apply(weights_init_normal)
        # self.classifiers_YN.apply(weights_init_normal)    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task = "YN",
        multi_task = False
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        #select the cls token
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

#         logits = []
        out_logits = []
        loss = 0
        # TODO check teh batch siZe
        # for ind, logit in enumerate(pooled_output): 
        if task == "YN":
            for ind  in range(self.num_labels_YN): 
                logit = self.classifiers_YN[ind](pooled_output)
                if labels is not None:
                    loss += self.criterion_YN[ind](logit, labels[:, ind])
                # logit = self.classifiers1[ind](pooled_output[ind])
#                 with torch.no_grad():
#                     logits.append(logit)
                out_logits.append(self.softmax(logit))
                # logit = self.classifiers1[ind](pooled_output[ind])

        elif task == "FR":
            for ind  in range(self.num_labels_FR): 
                logit = self.classifiers_FR[ind](pooled_output)
                if labels is not None:
                    loss += self.criterion_FR[ind](logit, labels[:, ind])
                # logit = self.classifiers1[ind](pooled_output[ind])
                # with torch.no_grad():
                # logits.append(logit)
                out_logits.append(self.softmax(logit))
    #             logits.append(logit.squeeze(0))

        
        if labels is not None:
            outputs = (loss, torch.stack(out_logits),) + outputs[2:]

        else:
            outputs = (None,torch.stack( out_logits),) + outputs[2:]

        return outputs  # (loss), reshaped_logits, (hidden_st



class BertMultiTaskBooleanQuestionLoad(BertPreTrainedModel):
    
    def __init__(self, config, 
                device = 'cuda:0', 
                no_dropout = False, 
                num_labels_YN = None, 
                num_labels_FR = None,  
                dataset = "spartqa",
                LM = "bert",
                has_batch = False,
                criterion = "focal"
                ):
        super().__init__(config)
        
        if no_dropout:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
            
        self.device1 = device

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes = 2
        self.num_labels_YN = num_labels_YN
        self.num_labels_FR = num_labels_FR
        # self.qtype = qtype

        if self.num_labels_YN:
            if dataset == "spartqa":
                self.alphasYN = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.27, 0.73] ]).to(self.device1)
            elif dataset == "human":
                self.alphasYN = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.1, 0.9] ]).to(self.device1)
            elif dataset == "spartun":
                self.alphasYN = torch.tensor([[0.55, 0.45], [0.45, 0.55]]).to(self.device1)
            else:
                self.alphasYN = torch.tensor([[0.5, 0.5]]*self.num_labels_YN).to(self.device1)
            
            #initialize the classifier model
            classifiers_YN = []
            self.criterion_YN = []
            for item in range(self.num_labels_YN):
                classifiers_YN.append(nn.Linear(config.hidden_size, self.num_classes))
                
                if criterion == "cross":
                    self.criterion_YN.append(nn.CrossEntropyLoss(weight=self.alphasYN[item]))
                else:
                    self.criterion_YN.append(FocalLoss(alpha=self.alphasYN[item], class_num=self.num_classes, gamma = 2))
            self.classifiers_YN_load = nn.ModuleList(classifiers_YN)
#             self.sigma_YN = nn.Parameter(torch.ones(1))


        if self.num_labels_FR:
            if dataset == "spartqa":

                self.alphasFR = torch.tensor([[0.20, 0.8], [0.20, 0.8], [0.25, 0.75], [0.4, 0.6], [0.1, 0.9], [0.02, 0.98], [0.02, 0.98]]).to(self.device1)
            elif dataset == "human":
                self.alphasFR = torch.tensor([[0.35, 0.65], [0.25, 0.75], [0.25, 0.75], [0.2, 0.8], [0.25, 0.75], [0.45, 0.55], [0.05, 0.95]]).to(self.device1)
            elif dataset == "spartun":
                self.alphasFR = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.27, 0.73], [0.26, 0.74], [0.1, 0.9], [0.1, 0.9], [0.02, 0.98], [0.15, 0.85], [0.26, 0.74], [0.07, 0.93], [0.002, 0.998], [0.02, 0.98], [0.1, 0.9], [0.03, 0.97], [0.08, 0.92]]).to(self.device1)
            elif dataset == "babi":
                self.alphasFR = torch.tensor([[0.60, 0.4], [0.57, 0.43], [0.6, 0.4], [0.41, 0.59]]).to(self.device1)
            else:
                self.alphasFR = torch.tensor([[0.5, 0.5]]*self.num_labels_FR)



            classifiers_FR = []
            self.criterion_FR = []
            #TODO add batch
            #TODO changed

            for item in range(self.num_labels_FR):
                classifiers_FR.append(nn.Linear(config.hidden_size, self.num_classes))
                if criterion == "corss":
                    self.criterion_FR.append(nn.CrossEntropyLoss(weight=self.alphasFR[item]))
                else:
                    self.criterion_FR.append(FocalLoss(alpha=self.alphasFR[item], class_num=self.num_classes, gamma = 2, size_average=True))

            self.classifiers_FR_load = nn.ModuleList(classifiers_FR)
                # self.sigma_FR = nn.Parameter(torch.ones(1))
        
        # self.sigma = nn.Parameter(torch.ones(2))
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        
        # self.classifiers_FR.apply(weights_init_normal)
        # self.classifiers_YN.apply(weights_init_normal)    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task = "YN",
        multi_task = False
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        #select the cls token
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

#         logits = []
        out_logits = []
        loss = 0
        # TODO check teh batch siZe
        # for ind, logit in enumerate(pooled_output): 
        if task == "YN":
            for ind  in range(self.num_labels_YN): 
                logit = self.classifiers_YN_load[ind](pooled_output)
                if labels is not None:
                    loss += self.criterion_YN[ind](logit, labels[:, ind])
                # logit = self.classifiers1[ind](pooled_output[ind])
#                 with torch.no_grad():
#                     logits.append(logit)
                out_logits.append(self.softmax(logit))
                # logit = self.classifiers1[ind](pooled_output[ind])

        elif task == "FR":
            for ind  in range(self.num_labels_FR): 
                logit = self.classifiers_FR_load[ind](pooled_output)
                if labels is not None:
                    loss += self.criterion_FR[ind](logit, labels[:, ind])
                # logit = self.classifiers1[ind](pooled_output[ind])
                # with torch.no_grad():
                # logits.append(logit)
                out_logits.append(self.softmax(logit))
    #             logits.append(logit.squeeze(0))

        
        if labels is not None:
            outputs = (loss, torch.stack(out_logits),) + outputs[2:]

        else:
            outputs = (None,torch.stack( out_logits),) + outputs[2:]

        return outputs  # (loss), reshaped_logits, (hidden_s



class BertForSpatialRelationExtraction(BertPreTrainedModel):
    def __init__(self, config, num_labels = 12,  device = 'cuda:0', no_dropout = False):
        super().__init__(config)
        if no_dropout:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
            
#         self.device = device
        self.num_labels = num_labels
        self.num_classes = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
#         self.classifier1 = nn.Linear(config.hidden_size, self.num_labels)
#         self.classifier2 = nn.Linear(config.hidden_size, self.num_type_class)
        
#         self.classifiers = nn.ModuleList([self.classifier1, self.classifier2])

        self.alphas = torch.tensor([[0.5, 0.5]]*self.num_labels_FR)
        classifiers = []
        self.criterion = []

        for item in range(self.num_labels):
            classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
#             if criterion == "corss":
#                 self.criterion.append(nn.CrossEntropyLoss(weight=self.alphasFR[item]))
#             else:
            self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2, size_average=True))

        self.classifiers = nn.ModuleList(classifiers)
            
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        loss = 0
        out_logits = []
        for ind  in range(self.num_labels_FR): 
            logit = self.classifiers[ind](pooled_output)
            if labels is not None:
                loss += self.criterion[ind](logit, labels[:, ind])
            out_logits.append(self.softmax(logit))

        if labels is not None:
            outputs = (loss, torch.stack(out_logits),) + outputs[2:]

        else:
            outputs = (None,torch.stack( out_logits),) + outputs[2:]

        return outputs  # (loss), reshaped_logits, (hidden_s

#         if labels is not None:
            
# #             label1 = labels[0].float()
#             if self.num_labels == 1:
#                 #  We are doing regression
# #                 loss_fct = MSELoss()
#                 loss_fct = BCELoss()
#                 out_logits1 = self.sigmoid(logits1)
#                 loss += loss_fct(out_logits1.view(-1), labels[0].view(-1))                
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss += loss_fct(logits1.view(-1, self.num_labels), labels[0].view(-1))
#                 out_logits1 = logits1#self.softmax(logits1)
# #             outputs1 = (loss,) + outputs1
            
    
# #             label2 = labels[1].long()
#             if self.num_type_class == 1:
#                 #  We are doing regression
# #                 loss_fct = MSELoss()
#                 out_logits2 = self.sigmoid(logits2)
#                 loss_fct = BCELoss()
#                 loss += loss_fct(out_logits2.view(-1), labels[1].view(-1))
                
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss += loss_fct(logits2.view(-1, self.num_type_class), labels[1].view(-1))
#                 out_logits2 = logits2#self.softmax(logits2)
# #             outputs2 = (loss,) + outputs2
        
#         else:
            
#             out_logits1 = self.sigmoid(logits1)
#             out_logits2 = logits2
            
#         outputs1 = (out_logits1,) + outputs[2:]  # add hidden states and attention if they are here
#         outputs2 = (out_logits2,) + outputs[2:]
        
#         return loss, outputs1, outputs2  # (loss), logits, (hidden_states), (attentions)    
    
    
class BertForSequenceClassification1(BertPreTrainedModel):
    def __init__(self, config, type_class = 0,  device = 'cuda:0', no_dropout = False):
        super().__init__(config)
        if no_dropout:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
            
#         self.device = device
        self.num_labels = config.num_labels
        self.num_type_class = type_class
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, self.num_type_class)
        
#         self.classifiers = nn.ModuleList([self.classifier1, self.classifier2])
        self.init_weights()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)

        
        loss = 0
        if labels is not None:
            
#             label1 = labels[0].float()
            if self.num_labels == 1:
                #  We are doing regression
#                 loss_fct = MSELoss()
                loss_fct = BCELoss()
                out_logits1 = self.sigmoid(logits1)
                loss += loss_fct(out_logits1.view(-1), labels[0].view(-1))                
            else:
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(logits1.view(-1, self.num_labels), labels[0].view(-1))
                out_logits1 = logits1#self.softmax(logits1)
#             outputs1 = (loss,) + outputs1
            
    
#             label2 = labels[1].long()
            if self.num_type_class == 1:
                #  We are doing regression
#                 loss_fct = MSELoss()
                out_logits2 = self.sigmoid(logits2)
                loss_fct = BCELoss()
                loss += loss_fct(out_logits2.view(-1), labels[1].view(-1))
                
            else:
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(logits2.view(-1, self.num_type_class), labels[1].view(-1))
                out_logits2 = logits2#self.softmax(logits2)
#             outputs2 = (loss,) + outputs2
        
        else:
            
            out_logits1 = self.sigmoid(logits1)
            out_logits2 = logits2
            
        outputs1 = (out_logits1,) + outputs[2:]  # add hidden states and attention if they are here
        outputs2 = (out_logits2,) + outputs[2:]
        
        return loss, outputs1, outputs2  # (loss), logits, (hidden_states), (attentions)    
    
# class BertForBooleanQuestionFB(BertPreTrainedModel):
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
        
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
            
#         self.device2 = device
#         self.bert = BertModel(config)
#         self.bert_answer = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2

#         self.classifier = nn.Linear(config.hidden_size, self.num_classes)
#         self.alpha = torch.tensor([0.5, 0.5]).to(self.device2)
#         self.criterion = FocalLoss(alpha=self.alpha, class_num=self.num_classes, gamma = 2)
#         self.rnn = nn.LSTM(config.hidden_size, int(config.hidden_size/2), 1, bidirectional=True)
#         self.l1 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.l2 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         options=None
#     ):

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )


#         pooled_output = outputs[0]
        
#         pooled_output = self.dropout(pooled_output)
#         pooled_output, _ = self.rnn(pooled_output)
#         pooled_output = torch.stack([pooled[-1] for pooled in pooled_output])
#         logits = self.classifier(pooled_output)

#         if labels is not None:
#             loss = self.criterion(logits, labels)
# #             print(loss)
#             out_logits = self.softmax(logits)
            
#             outputs = (loss, out_logits,) + outputs[2:]
# #             outputs = (,) + outputs

#         return outputs  
    

# class BertForBooleanQuestionCO(BertPreTrainedModel):
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
            
#         self.device2 = device
#         self.bert = BertModel(config)
#         self.bert_answer = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2

#         self.classifier = nn.Linear(config.hidden_size, self.num_classes)
#         self.alpha = torch.tensor([0.35, 0.65]).to(self.device2)
#         self.criterion = FocalLoss(alpha=self.alpha, class_num=self.num_classes, gamma = 2)
#         self.rnn = nn.LSTM(config.hidden_size, int(config.hidden_size/2), 1, bidirectional=True)
#         self.l1 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.l2 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         options=None
#     ):

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )


#         pooled_output = outputs[0]
#         pooled_output = self.dropout(pooled_output)
#         pooled_output, _ = self.rnn(pooled_output)
#         pooled_output = torch.stack([pooled[-1] for pooled in pooled_output])

        
        
#         logits = self.classifier(pooled_output)

#         if labels is not None:

#             loss = self.criterion(logits, labels)
#             out_logits = self.softmax(logits)

            
#             outputs = (loss, out_logits,) + outputs[2:]

#         return outputs  

# class BertForBooleanQuestionBabi(BertPreTrainedModel):
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
        
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
        
#         self.device = device
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2

#         self.classifier = nn.Linear(config.hidden_size, self.num_classes)
#         self.alpha = torch.tensor([0.5, 0.5]).to(self.device)
#         self.criterion = FocalLoss(alpha=self.alpha, class_num=self.num_classes, gamma = 2)
#         self.criterion = CrossEntropyLoss()
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )
#         pooled_output = outputs[1]
# #         print('pool out: ', pooled_output)
        
#         pooled_output = self.dropout(pooled_output)
# #         print(pooled_output.shape)
#         logits = self.classifier(pooled_output)
# #         print('logit tu', logits)

#         if labels is not None:
#             loss = self.criterion(logits, labels)
#             logits = self.softmax(logits)
#             outputs = (loss, logits,) + outputs[2:]

#         return outputs  # (loss), reshaped_logits, (hidden_st
    

    
# class BertForBooleanQuestionYN(BertPreTrainedModel):
    
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)

#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
        
#         self.device2 = device
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2

#         self.classifier = nn.Linear(config.hidden_size, self.num_classes)
# #         self.alphas = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.27, 0.73] ]).to(self.device)
# #         self.alphas = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5] ]).to('cuda:3')
# #         self.alphas = torch.tensor([[0.5, 0.5], [0.5, 0.5], [1, 0] ]).to(self.device)
#         self.alphas = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.27, 0.73] ]).to(self.device2)
#         classifiers = []
#         self.criterion = []
#         for item in range(3):
#             classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
#             self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
#         self.classifiers = nn.ModuleList(classifiers)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]

        
#         pooled_output = self.dropout(pooled_output)

#         logits = []
        
#         for ind in range(3): 
#             logit = self.classifiers[ind](pooled_output)

#             logits.append(logit.squeeze(0))

#         if labels is not None:

#             loss = 0
#             out_logits = []
#             for ind, logit in enumerate(logits):
#                 loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
#                 out_logits.append(self.softmax(logit))
#             outputs = (loss, torch.stack(out_logits),) + outputs[2:]

#         return outputs 


# class BertForBooleanQuestionYN1(BertPreTrainedModel):
    
#     def __init__(self,config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
        
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0

#         self.device1 = device
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2
#         self.num_labels = 3
#         self.alphas = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.27, 0.73] ]).to(self.device1)
#         # self.alphas = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).to(self.device)
        
#         classifiers = []
#         self.criterion1 = []
#         for item in range(self.num_labels):
#             classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
#             self.criterion1.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
#         self.classifiers1 = nn.ModuleList(classifiers)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
#         self.init_weights()
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )
#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output)
#         logits = []
#         for ind in range(self.num_labels): 
#             logit = self.classifiers1[ind](pooled_output)
#             logits.append(logit.squeeze(0))
#         if labels is not None:
#             loss = 0
#             out_logits = []
#             for ind, logit in enumerate(logits):
#                 loss += self.criterion1[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
#                 out_logits.append(self.softmax(logit))
#             outputs = (loss, torch.stack(out_logits),) + outputs[2:]
#         return outputs  


# class BertForBooleanQuestionCO1(BertPreTrainedModel):
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
            
#         self.device1 = device
#         self.bert = BertModel(config)
#         self.bert_answer = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2

#         self.classifier1 = nn.Linear(config.hidden_size, self.num_classes)
#         self.alpha = torch.tensor([0.35, 0.65]).to(self.device1)
#         self.criterion = FocalLoss(alpha=self.alpha, class_num=self.num_classes, gamma = 2)
#         self.rnn = nn.LSTM(config.hidden_size, int(config.hidden_size/2), 1, bidirectional=True)
#         self.l1 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.l2 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         options=None
#     ):

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )


#         pooled_output = outputs[0]
#         pooled_output = self.dropout(pooled_output)
#         pooled_output, _ = self.rnn(pooled_output)
#         pooled_output = torch.stack([pooled[-1] for pooled in pooled_output])

        
        
#         logits = self.classifier1(pooled_output)

#         if labels is not None:

#             loss = self.criterion(logits, labels)
#             out_logits = self.softmax(logits)

            
#             outputs = (loss, out_logits,) + outputs[2:]

#         return outputs  

# class BertForBooleanQuestionFB1(BertPreTrainedModel):
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
        
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
            
#         self.device1 = device
#         self.bert = BertModel(config)
#         self.bert_answer = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2

#         self.classifier1 = nn.Linear(config.hidden_size, self.num_classes)
#         self.alpha = torch.tensor([0.5, 0.5]).to(self.device1)
#         self.criterion = FocalLoss(alpha=self.alpha, class_num=self.num_classes, gamma = 2)
#         self.rnn = nn.LSTM(config.hidden_size, int(config.hidden_size/2), 1, bidirectional=True)
#         self.l1 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.l2 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         options=None
#     ):

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )


#         pooled_output = outputs[0]
        
#         pooled_output = self.dropout(pooled_output)
#         pooled_output, _ = self.rnn(pooled_output)
#         pooled_output = torch.stack([pooled[-1] for pooled in pooled_output])
#         logits = self.classifier1(pooled_output)

#         if labels is not None:
#             loss = self.criterion(logits, labels)
# #             print(loss)
#             out_logits = self.softmax(logits)
            
#             outputs = (loss, out_logits,) + outputs[2:]
# #             outputs = (,) + outputs

#         return outputs  

# class BertForBooleanQuestionFR1(BertPreTrainedModel):
    
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
        
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
            
#         self.device1 = device
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2

#         #self.classifier = nn.Linear(config.hidden_size, self.num_classes)
#         self.alphas = torch.tensor([[0.20, 0.8], [0.20, 0.8], [0.25, 0.75], [0.4, 0.6], [0.1, 0.9], [0.2, 0.98], [0.2, 0.98]]).to(self.device1)
#         classifiers = []
#         self.criterion = []
#         for item in range(7):
#             classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
#             self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
#         self.classifiers1 = nn.ModuleList(classifiers)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )
        
#         pooled_output = outputs[1]

        
#         pooled_output = self.dropout(pooled_output)
# #         print(pooled_output.shape)
#         logits = []
    
#         for ind, logit in enumerate(pooled_output): 
#             logit = self.classifiers1[ind](pooled_output[ind])
#             logits.append(logit)
        
#         # for check on YN
# #         for ind in range(7): 
# #             logit = self.classifiers[ind](pooled_output)
# #             logits.append(logit)
# #         print("FR",logits)

#         if labels is not None:

#             loss = 0
#             out_logits = []
#             for ind, logit in enumerate(logits):

#                 loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
#                 out_logits.append(self.softmax(logit))
#             outputs = (loss, torch.stack(out_logits),) + outputs[2:]

#         return outputs  # (loss), reshaped_logits, (hidden_st

# class BertForBooleanQuestionYNboolq(BertPreTrainedModel):
    
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)

#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
        
#         self.device1 = device
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2

#         self.classifier = nn.Linear(config.hidden_size, self.num_classes)
# #         self.alphas = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.27, 0.73] ]).to(self.device)
# #         self.alphas = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5] ]).to('cuda:3')
#         self.alphas = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).to(self.device1)
#         classifiers = []
#         self.criterion = []
#         for item in range(2):
#             classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
#             self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
#         self.classifiers = nn.ModuleList(classifiers)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]

        
#         pooled_output = self.dropout(pooled_output)

#         logits = []
        
#         for ind in range(2): 
#             logit = self.classifiers[ind](pooled_output)

#             logits.append(logit.squeeze(0))

#         if labels is not None:

#             loss = 0
#             out_logits = []
#             for ind, logit in enumerate(logits):
#                 loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
#                 out_logits.append(self.softmax(logit))
#             outputs = (loss, torch.stack(out_logits),) + outputs[2:]

#         return outputs 
    
# class BertForBooleanQuestionYNsprlqa(BertPreTrainedModel):
    
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)

#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
        
#         self.device1 = device
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2
#         self.num_labels = 2

#         self.classifier = nn.Linear(config.hidden_size, self.num_classes)
#         self.alphas = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).to(self.device1)
#         classifiers = []
#         self.criterion = []
#         for item in range(self.num_labels):
#             classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
#             self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
#         self.classifiers = nn.ModuleList(classifiers)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]

        
#         pooled_output = self.dropout(pooled_output)

#         logits = []
        
#         for ind in range(self.num_labels): 
#             logit = self.classifiers[ind](pooled_output)

#             logits.append(logit.squeeze(0))

#         if labels is not None:

#             loss = 0
#             out_logits = []
#             for ind, logit in enumerate(logits):
#                 loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
#                 out_logits.append(self.softmax(logit))
#             outputs = (loss, torch.stack(out_logits),) + outputs[2:]

#         return outputs 

# class BertForBooleanQuestionYNsprlqaLoad(BertPreTrainedModel):
    
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)

#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
        
#         self.device1 = device
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2
#         self.num_labels = 2

#         self.classifier1 = nn.Linear(config.hidden_size, self.num_classes)
#         self.alphas = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).to(self.device1)
#         classifiers = []
#         self.criterion1 = []
#         for item in range(self.num_labels):
#             classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
#             self.criterion1.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
#         self.classifiers1 = nn.ModuleList(classifiers)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax1 = nn.Softmax()
        
#         self.init_weights()
        

#     #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]

        
#         pooled_output = self.dropout(pooled_output)

#         logits = []
        
#         for ind in range(self.num_labels): 
#             logit = self.classifiers1[ind](pooled_output)

#             logits.append(logit.squeeze(0))

#         if labels is not None:

#             loss = 0
#             out_logits = []
#             for ind, logit in enumerate(logits):
#                 loss += self.criterion1[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
#                 out_logits.append(self.softmax1(logit))
#             outputs = (loss, torch.stack(out_logits),) + outputs[2:]

#         return outputs 

# class BertForBooleanQuestionFRsprlqa(BertPreTrainedModel):
    
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
        
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
            
#         self.device1 = device
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2
        
        
#         self.classifier = nn.Linear(config.hidden_size, self.num_classes)
# #         self.alphas = torch.tensor([[0.20, 0.8], [0.20, 0.8], [0.25, 0.75], [0.4, 0.6], [0.1, 0.9], [0.2, 0.98], [0.2, 0.98]]).to(self.device2)
#         self.alphas = torch.tensor([[0.09, 0.91], [0.44, 0.56], [0.007, 0.993], [0.007, 0.993], [0.086, 0.914], [0.057, 0.943], [0.013, 0.987], [0.13, 0.87], [0.03, 0.97], [0.006, 0.994], [0.13, 0.87], [0.007, 0.993], [0.0015, 0.9985], [0.013, 0.987], [0.003, 0.997], [0.003, 0.997], [0.006, 0.994], [0.003, 0.997]]).to(self.device1)
#         classifiers = []
#         self.criterion = []
#         for item in range(18):
#             classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
#             self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
#         self.classifiers = nn.ModuleList(classifiers)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )
        
#         pooled_output = outputs[1]

        
#         pooled_output = self.dropout(pooled_output)
# #         print(pooled_output.shape)
#         logits = []
    
#         for ind, logit in enumerate(pooled_output): 
#             logit = self.classifiers[ind](pooled_output[ind])
#             logits.append(logit)
        
#         # for check on YN
# #         for ind in range(7): 
# #             logit = self.classifiers[ind](pooled_output)
# #             logits.append(logit)
# #         print("FR",logits)

#         if labels is not None:

#             loss = 0
#             out_logits = []
#             for ind, logit in enumerate(logits):

#                 loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
#                 out_logits.append(self.softmax(logit))
#             outputs = (loss, torch.stack(out_logits),) + outputs[2:]

#         return outputs 
    
# class BertForBooleanQuestion3ClassYN(BertPreTrainedModel):
#     def __init__(self, config, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)

#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
        
#         self.device = device
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.num_classes = 2

#         self.classifier = nn.Linear(config.hidden_size, self.num_classes)
#         self.alphas = torch.tensor([[0.5, 0.5], [0.5, 0.5], [1, 0] ]).to(self.device)
#         classifiers = []
#         self.criterion = []
#         for item in range(3):
#             classifiers.append(nn.Linear(config.hidden_size, self.num_classes))
#             self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
#         self.classifiers = nn.ModuleList(classifiers)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
        
#         self.init_weights()
        

#     #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]

        
#         pooled_output = self.dropout(pooled_output)
# #         print('$',pooled_output.shape)
#         logits = []
# #         for ind, logit in enumerate(pooled_output): 
# #             logit = self.classifiers[ind](pooled_output[ind])
# #             logits.append(logit)
#         for ind in range(3): 
#             logit = self.classifiers[ind](pooled_output)
# #             print("#", logit.squeeze(0).shape)
#             logits.append(logit.squeeze(0))

#         if labels is not None:

#             loss = 0
#             out_logits = []
#             for ind, logit in enumerate(logits):
# #                 weights = torch.ones(2).float()
# #                 alpha = self.alphas[ind]
# #                 print("**",labels.shape ,labels[ind], labels[ind].unsqueeze(0))
# #                 print("**",logit.shape)
#                 loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
#                 out_logits.append(self.softmax(logit))
#             outputs = (loss, torch.stack(out_logits),) + outputs[2:]
# #             outputs = (,) + outputs




#         return outputs  # (loss), reshaped_logits, (hidden_st

# class BertForSequenceClassification1(BertPreTrainedModel):
#     def __init__(self, config, type_class = 0,  device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
            
# #         self.device = device
#         self.num_labels = config.num_labels
#         self.num_type_class = type_class
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
#         self.classifier1 = nn.Linear(config.hidden_size, self.num_labels)
#         self.classifier2 = nn.Linear(config.hidden_size, self.num_type_class)
        
# #         self.classifiers = nn.ModuleList([self.classifier1, self.classifier2])
#         self.init_weights()
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
        
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output)
        
#         logits1 = self.classifier1(pooled_output)
#         logits2 = self.classifier2(pooled_output)

        
#         loss = 0
#         if labels is not None:
            
# #             label1 = labels[0].float()
#             if self.num_labels == 1:
#                 #  We are doing regression
# #                 loss_fct = MSELoss()
#                 loss_fct = BCELoss()
#                 out_logits1 = self.sigmoid(logits1)
#                 loss += loss_fct(out_logits1.view(-1), labels[0].view(-1))                
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss += loss_fct(logits1.view(-1, self.num_labels), labels[0].view(-1))
#                 out_logits1 = logits1#self.softmax(logits1)
# #             outputs1 = (loss,) + outputs1
            
    
# #             label2 = labels[1].long()
#             if self.num_type_class == 1:
#                 #  We are doing regression
# #                 loss_fct = MSELoss()
#                 out_logits2 = self.sigmoid(logits2)
#                 loss_fct = BCELoss()
#                 loss += loss_fct(out_logits2.view(-1), labels[1].view(-1))
                
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss += loss_fct(logits2.view(-1, self.num_type_class), labels[1].view(-1))
#                 out_logits2 = logits2#self.softmax(logits2)
# #             outputs2 = (loss,) + outputs2
        
#         else:
            
#             out_logits1 = self.sigmoid(logits1)
#             out_logits2 = logits2
            
#         outputs1 = (out_logits1,) + outputs[2:]  # add hidden states and attention if they are here
#         outputs2 = (out_logits2,) + outputs[2:]
        
#         return loss, outputs1, outputs2  # (loss), logits, (hidden_states), (attentions)

# class BertForSequenceClassification2(BertPreTrainedModel):
#     def __init__(self, config, type_class = 0,  device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
            
# #         self.device = device
#         self.num_labels = config.num_labels
#         self.num_type_class1 = type_class
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
#         self.classifier1 = nn.Linear(config.hidden_size, self.num_labels)
#         self.classifier21 = nn.Linear(config.hidden_size, self.num_type_class1)
# #         self.classifiers = nn.ModuleList([self.classifier1, self.classifier2])
#         self.init_weights()
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
        
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output)
        
#         logits1 = self.classifier1(pooled_output)
#         logits2 = self.classifier21(pooled_output)

        
#         loss = 0
#         if labels is not None:
            
# #             label1 = labels[0].float()
#             if self.num_labels == 1:
#                 #  We are doing regression
# #                 loss_fct = MSELoss()
#                 loss_fct = BCELoss()
#                 out_logits1 = self.sigmoid(logits1)
#                 loss += loss_fct(out_logits1.view(-1), labels[0].view(-1))                
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss += loss_fct(logits1.view(-1, self.num_labels), labels[0].view(-1))
#                 out_logits1 = logits1#self.softmax(logits1)
# #             outputs1 = (loss,) + outputs1
            
    
# #             label2 = labels[1].long()
#             if self.num_type_class1 == 1:
#                 #  We are doing regression
# #                 loss_fct = MSELoss()
#                 out_logits2 = self.sigmoid(logits2)
#                 loss_fct = BCELoss()
#                 loss += loss_fct(out_logits2.view(-1), labels[1].view(-1))
                
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss += loss_fct(logits2.view(-1, self.num_type_class1), labels[1].view(-1))
#                 out_logits2 = logits2#self.softmax(logits2)
# #             outputs2 = (loss,) + outputs2
        
#         else:
            
#             out_logits1 = self.sigmoid(logits1)
#             out_logits2 = logits2
            
#         outputs1 = (out_logits1,) + outputs[2:]  # add hidden states and attention if they are here
#         outputs2 = (out_logits2,) + outputs[2:]
        
#         return loss, outputs1, outputs2  # (loss), logits, (hidden_states), (attentions)


# class BertForSequenceClassification3(BertPreTrainedModel):
#     def __init__(self, config, type_class = 0, num_asked_class = 1, device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
            
# #         self.device = device
#         self.num_labels = config.num_labels
#         self.num_type_class1 = type_class
#         self.num_asked_class = num_asked_class
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
#         self.classifier1 = nn.Linear(config.hidden_size, self.num_labels)
#         self.classifier21 = nn.Linear(config.hidden_size, self.num_type_class1)
#         # self.classifier3 = nn.Linear(config.hidden_size, self.num_asked_class)
# #         self.classifiers = nn.ModuleList([self.classifier1, self.classifier2])
#         self.init_weights()
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         asked_compute = None,
#     ):
        
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output)
        
#         logits1 = self.classifier1(pooled_output)
#         logits2 = self.classifier21(pooled_output)
#         # if asked_compute != None:
#         #     logits3 = self.classifier3(pooled_output)

        
#         loss = 0
#         if labels is not None:
            
# #             label1 = labels[0].float()
#             if self.num_labels == 1:
#                 #  We are doing regression
# #                 loss_fct = MSELoss()
#                 loss_fct = BCELoss()
#                 out_logits1 = self.sigmoid(logits1)
#                 loss += loss_fct(out_logits1.view(-1), labels[0].view(-1))                
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss += loss_fct(logits1.view(-1, self.num_labels), labels[0].view(-1))
#                 out_logits1 = logits1#self.softmax(logits1)
# #             outputs1 = (loss,) + outputs1
            
    
# #             label2 = labels[1].long()
#             if self.num_type_class1 == 1:
#                 #  We are doing regression
# #                 loss_fct = MSELoss()
#                 out_logits2 = self.sigmoid(logits2)
#                 loss_fct = BCELoss()
#                 loss += loss_fct(out_logits2.view(-1), labels[1].view(-1))
            
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss += loss_fct(logits2.view(-1, self.num_type_class1), labels[1].view(-1))
#                 out_logits2 = logits2#self.softmax(logits2)
# #             outputs2 = (loss,) + outputs2

#             # if asked_compute !=None and self.num_asked_class == 1:
#             #     loss_fct = BCELoss()
#             #     out_logits3 = self.sigmoid(logits3)
#             #     loss += loss_fct(out_logits3.view(-1), labels[0].view(-1))
#             # else: 
#             #     out_logits3 = None


#         else:
            
#             out_logits1 = self.sigmoid(logits1)
#             out_logits2 = logits2
#             # out_logits3 = logits3 if asked_compute != None else None

            
#         outputs1 = (out_logits1,) + outputs[2:]  # add hidden states and attention if they are here
#         outputs2 = (out_logits2,) + outputs[2:]
#         # outputs3 = (out_logits3,) + outputs[2:]

        
#         return loss, outputs1, outputs2#, outputs3  # (loss), logits, (hidden_states), (attentions)
    
# class BertForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config, type_class = 0,  device = 'cuda:0', no_dropout = False):
#         super().__init__(config)
#         self.num_labels = config.num_labels
        
#         if no_dropout:
#             config.hidden_dropout_prob = 0.0
#             config.attention_probs_dropout_prob = 0.0
        
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
#         self.sigmoid = nn.Sigmoid()
        
#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = BCELoss()
#                 loss = loss_fct(self.sigmoid(logits).view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)


