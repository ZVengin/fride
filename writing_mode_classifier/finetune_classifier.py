import json,optuna,logging,sys,torch,argparse,random,torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from transformers import BertTokenizer,BertForSequenceClassification
#from datasets import metric
from sklearn.metrics import precision_recall_fscore_support
import evaluate



#model_checkpoint = "bert-base-uncased"
#model_checkpoint = "xlnet-base-cased"
model_checkpoint = "roberta-base"
#model_checkpoint = "model/checkpoint-1000"
num_labels=4
#random.seed(123)
#seeds = random.sample(range(40),10)

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')
logger = logging.getLogger()

id2label={0:'Dialogue',1:'Action',2:'Description',3:'Unknown'}

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=num_labels)
    return model


def process_dataset(dataset,tokenizer):
    def preprocess_function(examples):
        return tokenizer(examples['target'],truncation=True)

    encoded_dataset=dataset.map(preprocess_function,batched=True)
    return encoded_dataset


def load_data(train_file=None,val_file=None,test_file=None, format='csv'):
    files ={}
    if train_file is not None:
        files['train'] = train_file
    if val_file is not None:
        files['validation']=val_file
    if test_file is not None:
        files['test']=test_file
    if train_file is None and val_file is None and test_file is None:
        raise Exception('you need to specify at one file!')
    dataset = load_dataset(format,data_files=files)
    return dataset


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def run_trainer(args, dataset, hyper_search=False, hyper_space=None, save_path='best_run.json', do_predict=False):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    encoded_dataset = process_dataset(dataset, tokenizer)
    trainer = Trainer(
        model_init=load_model,
        args=args,
        train_dataset=encoded_dataset['train'] if set_size != 0 else None,
        eval_dataset=encoded_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)
    if hyper_search:
        best_run=trainer.hyperparameter_search(hp_space=hyper_space)
        logger.info(f'find the best hyper parameters:{best_run}')
        with open(save_path,'w') as f:
            json.dump({'best_valid_score':best_run,'parameters':best_run.params},f)
    else:
        if set_size != 0:
            trainer.train()
    eval_result = trainer.evaluate()
    if not do_predict:
        return eval_result,None
    else:
        pred_outputs = trainer.predict(encoded_dataset['test'])
        pred_ids = pred_outputs.predictions.argmax(axis=1).tolist()
        if 'label' in dataset['test'].column_names:
            test_result = precision_recall_fscore_support(y_pred=pred_ids, y_true=dataset['test']['label'],
                                                     average='macro')
            test_result = {k:v for k,v in zip(['precision','recall','f1','support'],list(test_result))}
            logger.info(f'test score:{test_result}')
            logger.info(f'predict_ids:{pred_ids}')
            logger.info(f'target_ids:{dataset["test"]["label"]}')
    return eval_result,test_result



def run_finetune(trial=None, hyper_setting=None,model_name=None, do_predict=False):
    if model_name is None:
        model_name = specified_model_name
    if hyper_setting is None and trial is None:
        hyper_setting = {
            'learning_rate':2e-5,
            'per_device_train_batch_size':8,
            'num_train_epochs':5,
            'seed':random.randint(0,40)
        }
    elif hyper_setting is None and trial is not None:
        hyper_setting = {
            'learning_rate': trial.suggest_float('learning_rate',1e-5,5e-5),
            'per_device_train_batch_size': trial.suggest_int('per_device_train_batch_size',2,8),
            'num_train_epochs': 10,#trial.suggest_int('num_train_epochs',3,5),
            'seed': 27,#trial.suggest_int('seed',1,40)
        }
    logger.info(f'==>>> Loading training file from: {data_dir + train_set}')
    dataset = load_data((data_dir + train_set) if set_size != 0 else None,
                        data_dir + valid_set,
                        data_dir + test_set).shuffle()

    args = TrainingArguments(
        output_dir=f'./../exp/writing_mode_classifier/{model_name}{"" if trial is None else "-hyper-search"}',
        evaluation_strategy='epoch',
        learning_rate=hyper_setting['learning_rate'],
        per_device_train_batch_size=hyper_setting['per_device_train_batch_size'],
        per_device_eval_batch_size=4,
        num_train_epochs=10,#hyper_setting['num_train_epochs'],
        weight_decay=0.01,
        seed=hyper_setting['seed'],
        load_best_model_at_end=False,
        metric_for_best_model='f1')

    eval_result,test_result = run_trainer(args,dataset,do_predict=do_predict)
    logger.info(f'last validation result:{eval_result}')
    if test_result is None:
        return eval_result
    else:
        return eval_result,test_result


def run_trial(trial):
    #with open('temp.json','r') as f:
    #    set_names = json.load(f)
    #    train_set,valid_set,test_set = set_names['train_set'],set_names['valid_set'],set_names['test_set']
    #logger.info(f'==>>> Loading training file from: {data_dir + train_set}')
    #dataset = load_data(data_dir + train_set,
    #                    data_dir + valid_set,
    #                    data_dir + test_set)
    return run_finetune(trial=trial)['eval_f1']


def run_hyper_search():
    data_dir = './dataset/'
    model_name = model_checkpoint.split('/')[-1]
    args = TrainingArguments(
        output_dir=f'./../exp/writing_mode_classifier/{model_name}-hyper-search',
        evaluation_strategy='epoch',
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1')
    dataset = load_data(data_dir + 'train_set.csv',
                        data_dir + 'val_set.csv',
                        data_dir + 'test_set.csv')
    eval_result = run_trainer(args, dataset, hyper_search=True)

    return eval_result


def run_optuna_search(n_trials,study_name):
    optuna.logging.get_logger("writing-mode-classifier").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name,direction='maximize')
    study.optimize(run_trial, n_trials=n_trials)
    study_result = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    study_result.to_csv(result_dir+f'{study_name}-result.csv',index=False)


def load_best_hyper_setting(path):
    settings = pd.read_csv(path)
    best_setting = settings[settings.index==settings['value'].idxmax()].to_dict('records')[0]
    best_setting = {k.replace('params_',''):v for k,v in best_setting.items()}
    return best_setting
# {'learning_rate': 1.9167526807754088e-05, 'num_train_epochs': 5, 'seed': 3, 'per_device_train_batch_size': 8}.


def load_predict_model(model_checkpoint,out_dir=None):
    logger.info(f'==>>> cuda {"is" if torch.cuda.is_available() else "is not"} available.')
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels)#.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    args = TrainingArguments(
        output_dir=out_dir if out_dir is not None else f'./../exp/writing_mode_classifier/Bert-split3-100-trial0',
        per_device_eval_batch_size=16
    )
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer, compute_metrics=compute_metrics)
    return trainer

def predict_test_set(trainer,test_file, format='csv'):
    dataset = load_data(test_file=test_file,format=format)
    # print(dataset)
    encoded_dataset = process_dataset(dataset, trainer.tokenizer)
    pred_outputs = trainer.predict(encoded_dataset['test'])
    pred_ids = pred_outputs.predictions.argmax(axis=1).tolist()
    pred_labels = [id2label[i] for i in pred_ids]
    if 'label' in dataset['test'].column_names:
        scores = precision_recall_fscore_support(y_pred=pred_ids, y_true=dataset['test']['label'], average='weighted')
        test_result = {k: v for k, v in zip(['precision', 'recall', 'f1', 'support'], list(scores))}
        logger.info(f'test score:{test_result}')
        return pred_labels,test_result
    else:
        return pred_labels

def predict_sample(model,tokenizer,text):
    inputs = tokenizer(text,return_tensors='pt')
    #device = model.get_device()
    inputs = {k:v.to(torch.device(0)) for k,v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    predict_id = torch.topk(logits,1)[1][0].item()
    return id2label[predict_id]


def run_predict(model_checkpoint, test_file):
    trainer = load_predict_model(model_checkpoint)
    pred_labels,test_result = predict_test_set(trainer,test_file)
    return pred_labels,test_result


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('split_name', type=str)
    parser.add_argument('trial_name', type=str)
    parser.add_argument('train_size',type=str)
    return parser


def do_hyper_search(data_sizes):
    global train_set,set_size
    for set_size in data_sizes:
        train_set = train_set_temp.format(set_size)
        run_optuna_search(n_trials=20, study_name=study_name.format(args.split_name,set_size,args.trial_name))



def test_model(data_sizes):
    global train_set,set_size
    eval_scores, test_scores = [], []
    #seeds = random.sample(range(0,40),len(data_sizes))
    seeds=[27]
    for idx,set_size in enumerate(data_sizes):
        if set_size != 0:
            train_set = train_set_temp.format(set_size)
            best_setting = load_best_hyper_setting(
                result_dir+f'{study_name.format(args.split_name,set_size, "trial0")}-result.csv')
            best_setting['seed']= seeds[idx]
            eval_result, test_result = run_finetune(
                hyper_setting=best_setting,
                model_name=f'{model_checkpoint}-{args.split_name}-{set_size}-{args.trial_name}',
                do_predict=True)
        else:
            #_,eval_result = run_predict(model_checkpoint,data_dir+valid_set)
            #_,test_result = run_predict(model_checkpoint,data_dir+test_set)
            train_set = train_set_temp.format(set_size)
            best_setting = None
            eval_result, test_result = run_finetune(
                hyper_setting=best_setting,
                model_name=f'{model_checkpoint}-{args.split_name}-{set_size}-{args.trial_name}',
                do_predict=True)
        #eval_result,test_result=run_finetune(model_name=f'Bert-{args.split_name}-{set_size}-{args.trial_name}',do_predict=True)
        eval_result['data_size']=set_size
        test_result['data_size']=set_size
        test_result['random_seed']=best_setting['seed'] if best_setting != None else -1
        eval_scores.append(eval_result)
        test_scores.append(test_result)

    eval_scores_df = pd.DataFrame(data=eval_scores)
    eval_scores_df.to_csv(result_dir+f'eval_scores_{args.split_name}_{args.trial_name}.csv',index=False)
    test_scores_df = pd.DataFrame(data=test_scores)
    test_scores_df.to_csv(result_dir+f'test_scores_{args.split_name}_{args.trial_name}.csv',index=False)



#run_hyper_search()
cache_dir='/home/u00483/.cache/'
data_dir = './dataset3/'
result_dir='./test_result15/'
#data_sizes=[100,200,300,400,500,600,700,800,900,1000]
#data_sizes=[50]
#data_sizes=[50,100,150]
study_name = "writing-mode-classifier-study-{}-{}-{}"
train_set_temp = 'subtrain_set_{}.csv'
valid_set = 'valid_set.csv'
test_set = 'test_set.csv'
specified_model_name_temp=model_checkpoint.split('/')[-1]+'-{}-{}'
eval_scores,test_scores=[],[]
#train_set = train_set_temp.format(50)
#run_optuna_search(n_trials=50, study_name=study_name.format(50))
if __name__=='__main__':
    parser = get_parser()
    args=parser.parse_args()
    experiment_id = f'{args.split_name}-{args.trial_name}-{args.train_size}'
    metric = evaluate.load('f1')  # `experiment_id` is not a parameter in the new API
    #metric = load_metric('f1',experiment_id=f'{args.split_name}-{args.trial_name}-{args.train_size}')
    specified_model_name = specified_model_name_temp.format(args.split_name,args.trial_name)

    ##################
    data_sizes = [int(train_size) for train_size in args.train_size.split(',') if train_size.strip() and train_size.isdigit()]
    #do_hyper_search(data_sizes)
    test_model(data_sizes)

    ##################
    #for set_size in data_sizes:
    #    train_set = train_set_temp.format(set_size)
    #    run_optuna_search(n_trials=10, study_name=study_name.format(args.split_name,set_size,args.trial_name))
    #    #best_setting=load_best_hyper_setting(f'{study_name.format(args.split_name,set_size,args.trial_name)}-result.csv')
    #    best_setting = load_best_hyper_setting(result_dir+f'{study_name.format(args.split_name, set_size, "trial0")}-result.csv')
    #    best_setting['seed']= random.randint(0,40)
    #    eval_result,test_result=run_finetune(hyper_setting=best_setting,model_name=f'Bert-{args.split_name}-{set_size}-{args.trial_name}',do_predict=True)
    #    #eval_result,test_result=run_finetune(model_name=f'Bert-{args.split_name}-{set_size}-{args.trial_name}',do_predict=True)
    #    eval_result['data_size']=set_size
    #    test_result['data_size']=set_size
    #    eval_scores.append(eval_result)
    #    test_scores.append(test_result)

    #eval_scores_df = pd.DataFrame(data=eval_scores)
    #eval_scores_df.to_csv(result_dir+f'eval_scores_{args.split_name}_{args.trial_name}.csv',index=False)
    #test_scores_df = pd.DataFrame(data=test_scores)
    #test_scores_df.to_csv(result_dir+f'test_scores_{args.split_name}_{args.trial_name}.csv',index=False)

#run_finetune()

#model_path = './../exp/writing_mode_classifier/distilbert-base-uncased/checkpoint-165'
#test_file = './dataset/test_set.csv'
#run_predict(model_path,test_file)
