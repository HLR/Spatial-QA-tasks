
## **Baselines**
All qtypes can be cast into a sequence classification task, and the three transformer-based LMs tested in this paper, BERT , ALBERT, and XLNet, can all handle this type of tasks by classifying the representation of [CLS], a special token prepended to each target sequence. Depending on the qtype, the input sequence and how we do inference may be different.


For running each baseline At first you should install required packages. (torch, transformers (v 4.0.1)) to do this it is better to create a virtual environment and install all packages in there.

Then you should create and empty dataset directory, and create a directory for each dataset you want to test (SpaRTUN) and upload dataset files in it.

After all fo this you should add the related arguments to the running command.
The list of all arguments are listed below:

    "--research_hlr",help="change the location of files",action='store_true', default = True)
    "--result",help="Name of the result's saving file", type= str, default='test')
    "--result_folder",help="Name of the folder of the results file", type= str, default='transfer/Results')
    "--model",help="Name of the model's saving file", type= str, default='')
    "--model_folder",help="Name of the folder of the models file", type=str, default = "transfer/Models")
    "--old_experiments",help="from the spartun project some setting of models changes, so if you want to run the previous things, set this True", default = False, action='store_true')
    "--dataset",help="name of the dataset like mSpRL or spaceeval", type = str, default = 'spartqa')
    "--no_save",help="If save the model or not", action='store_true', default = False)
    "--load",help="For loading model", type=str)
    "--cuda",help="The index of cuda", type=int, default=None)
    "--qtype",help="Name of Question type. (FB, FR, CO, YN)", type=str, default = 'all')
    "--train10k",help="Train on 10k data for babi dataset", action='store_true', default = False)
    "--train1k",help="Train on 1k data for babi dataset", action='store_true', default = False)
    "--train24k",help="Train on 24k data", action='store_true', default = False)
    "--train100k",help="Train on 100k data", action='store_true', default = False)
    "--train500",help="Train on 500 data", action='store_true', default = False)
    "--unseentest",help="Test on unseen data", action='store_true', default = False)
    "--human",help="Train and Test on human data", action='store_true', default = False)
    "--humantest",help="Test on human data", action='store_true', default = False)
    "--dev_exists", help="If development set is used", action='store_true', default = False)
    "--test_track", help="track the test result during training", action='store_true', default = False)
    "--no_train",help="Number of train samples", action='store_true', default = False)
    "--save_data",help="save extracted data", action='store_true', default = False)
    "--baseline",help="Name of the baselines. Options are 'bert', 'xlnet', 'albert'", type=str, default = 'bert')
    "--pretrain",help="Name of the pretrained model. Options are 'bertqa', 'bertbc' (for bert boolean clasification), 'mlm', 'mlmr', 'tokencls'", type=str, default = 'bertbc')
    "--con",help="Testing consistency or contrast", type=str, default = 'not')
    "--optim",help="Type of optimizer. options 'sgd', 'adamw'.", type=str, default = 'adamw')
    "--loss",help="Type of loss function. options 'cross'.", type=str, default = 'focal')
    "--batch_size",help="size of batch. If none choose the whole example in one sample. If QA number of all questions if SIE number of sentences or triplets'.", type=int, default = 1)
    "--best_model",help="How to save the best model. based on aacuracy or f1 measure", type=str, default = 'accuracy')
    "--train",help="Number of train samples", type = int)
    "--train_log", help="save the log of train if true", default = False, action='store_true')
    "--start",help="The start number of train samples", type = int, default = 0)
    "--dev",help="Number of dev samples", type = int)
    "--test",help="Number of test samples", type = int)
    "--unseen",help="Number of unseen test samples", type = int)
    "--has_zero_eval", help="If True before starting the training have a test on the test set", default = False, action='store_true')
    "--stepgame_train_set",help="Number of sentence in stepgame dataset", type = str, default=None)
    "--stepgame_test_set",help="Number of sentence in stepgame dataset", type = str, default="1 2 3 4 5 6 7 8 9 10")
    "--epochs",help="Number of epochs for training", type = int, default=0)
    "--lr",help="learning rate", type = float, default=2e-6)
    "--dropout", help="If you want to set dropout=0", action='store_true', default = False)
    "--unfreeze", help="freeze the first layeres of the model except this numbers", type=int, default = 0)
    "--seed", help="set seed for reproducible result", type=int, default = 1)
    "--other_var",  dest='other_var', action='store', help="Other variable: classification (DK, noDK), random, fine-tune on unseen. for changing model load MLM from pre-trained model and replace other parts with new on", type=str)
    "--other_var2",  dest='other_var2', action='store', help="Other variable: classification (DK, noDK), random, fine-tune on unseen. for changing model load MLM from pre-trained model and replace other parts with new on", type=str)
    "--detail",help="a description about the model", type = str)
    "--options", help="describe the model features: 'q+s' + 'first_attention_stoq' + 'just_pass_entity'+ '2nd_attention_stoq'+ '2nd_attention_qtos' + ", type=str, default=None)
    "--top_k_sent", help="set top k for sentence", type=int, default=None)
    "--top_k_s", help="set top k for indicator, entity, and triplets: 3#4#3", type=str, default=None)
    "--top_k_q", help="set top k for indicator, entity, and triplets:  3#4#3", type=str, default=None)
    "--cls_input_dim", help="an integer based on the final input of boolean classification", type=int, default=768)


An example of a command is:

python3 main.py --dataset spartun --dev_exists --epochs 11 --batch_size 8 --lr 8e-6 --result FR-test --qtype FR --cuda 6 --epochs 11

if you want to train, dev or test on smaller data just set the number of arguments --train, --test, and --dev

Please check all addresses and change it based on your desire.

To change train, reading the data, or classes, you just need to change the "QA/train, QA/test, Create_LM_input_output.py, PLModels.py)