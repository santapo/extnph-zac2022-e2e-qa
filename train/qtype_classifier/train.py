import torch
from utils import *
from transformers import AdamW,get_linear_schedule_with_warmup
import time
from scripts.classify.metric import f1_score_func
import datetime
# from metric import classification_metric
import wandb
from tqdm import tqdm
import numpy as np

wandb.init(project="ZaloAI22 E2E QnA", name="init")

def trainer(**kwargs):
    args = kwargs['args']
    model = kwargs['model']
    train_dataloader = kwargs['train_dataloader']
    val_dataloader = kwargs['val_dataloader']
    tokenizer = kwargs['tokenizer']
    device = kwargs['device']

    print("start training")
    learning_rate = args.lr
    adam_epsilon = args.epsilon
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.2},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    num_epochs = args.epochs
    total_steps = len(train_dataloader) * num_epochs
    print(total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    num_mb_train = len(train_dataloader)
    num_mb_val = len(val_dataloader)
    print(num_mb_train)
    if num_mb_val == 0:
        num_mb_val = 1
    epochs = num_epochs
    best_acc = 0
    total_t0 = time.time()
    for epoch_i in range(1, num_epochs+1):

        t0 = time.time()
        total_loss = 0
        train_acc = 0
        predictions, true_train = [], []
        model.train()
        model.to(device)
        loss_train_total = 0

        progress_bar = tqdm(train_dataloader, desc='Epoch {:1d}'.format(epoch_i), leave=False, disable=False)


        for batch in progress_bar:

            model.zero_grad()
        
            batch = tuple(b.to(device) for b in batch)
        
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       

            outputs = model(**inputs)
        
            loss = outputs[0]
            loss_train_total += loss.item()
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_train.append(label_ids)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            loss_train_avg = loss_train_total/len(train_dataloader)   

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            wandb.log({"train_loss": loss.item()})

        predictions = np.concatenate(predictions, axis=0)
        true_train = np.concatenate(true_train, axis=0)
        train_f1 = f1_score_func(predictions, true_train)
        wandb.log({"train_acc":train_f1})
 
        model.eval()
    
        loss_val_total = 0
        predictions, true_vals = [], []
        
        for batch in val_dataloader:
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }

            with torch.no_grad():        
                outputs = model(**inputs)
                
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
        
        loss_val_avg = loss_val_total/len(val_dataloader) 
        
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        val_f1 = f1_score_func(predictions, true_vals)
        
        wandb.log({"val_loss": loss_val_avg})
        wandb.log({"val_acc":val_f1})


        #     nb_eval_steps += 1
        # avg_val_loss = eval_loss / len(val_dataloader)
        # avg_val_accuracy = eval_accuracy/nb_eval_steps
        # avg_val_precision = eval_precision/nb_eval_steps
        # avg_val_recall = eval_recall/nb_eval_steps
        # avg_val_f1 = eval_f1/nb_eval_steps
        
        # print("  Valid Loss: {0:.2f}".format(avg_val_loss))
        # print("  Acc score: {0:.2f}".format(avg_val_accuracy))
        # print("  Precision score: {0:.2f}".format(avg_val_precision))
        # print("  Recall score: {0:.2f}".format(avg_val_recall))
        # print("  F1 score: {0:.2f}".format(avg_val_f1))
        # print("  Validation took: {:}".format(format_time(time.time() - t0)))
        
        if val_f1 >=  best_acc:

        # output_dir = args.model_save + "model_" + str(epoch_i+1)
        # old_output_dir = args.model_save + "model_" + str(epoch_i) 
        # # if os.path.exists(old_output_dir):
        # #     shutil.rmtree(old_output_dir)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # print("Saving model to %s" % output_dir)

            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(args.model_save)
            tokenizer.save_pretrained(args.model_save)
        # # best_acc = avg_val_accuracy
