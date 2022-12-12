import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)


class Classify:
    def __init__(self,model_path,device,max_len ,batch_size,num_label):
        self.max_len = max_len
        self.batch_size = batch_size
        self.model_path = model_path
        self.num_label = num_label
        if device!="cpu" and device!="cuda":
            self.device = torch.device(self.get_device())
        else:
            self.device = torch.device(device)
        self.load_model()
    
    def get_device(self):
        if torch.cuda.is_available():       
            device_name = "cuda"

            print('There are %d GPU(s) available.' % torch.cuda.device_count())

            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device_name = "cpu"
        return device_name

    def load_model(self):
        print('Loading BERT tokenizer...')
        self.tokenizer =  AutoTokenizer.from_pretrained(self.model_path, do_lower_case=False)

        self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels = self.num_label,
                output_attentions = False,
                output_hidden_states = False,
        ).to(self.device)
        self.model.eval()

    def predict_text(self,texts):
        #1 comment for 1 runtime

        y_preds = []
        prob_preds = []
        encoded_data = self.tokenizer.batch_encode_plus(
                                texts, 
                                add_special_tokens=True, 
                                return_attention_mask=True, 
                                pad_to_max_length=True, 
                                max_length=self.max_len,
                                truncation=True,
                                return_tensors='pt'
                            )
        input_ids = encoded_data['input_ids'].to(self.device)
        attention_masks = encoded_data['attention_mask'].to(self.device)


        outputs = self.model(input_ids,attention_mask=attention_masks)[0]

        output = torch.sigmoid(outputs)
        output = output.cpu().detach().numpy().tolist()

        b_preds = [np.argmax(b_output).item() for b_output in output]
        
        # text_label += b_preds
        # text_prob += [output[i][b_preds[i]] for i in range(len(output)) ]
        # print(text_label,text_prob)
        #     if 0 in text_label:
        #         y_preds.append(0)
        #     else:
        #         y_preds.append(1)
        #     prob_preds.append(sum(text_prob)/len(text_prob))
        # assert len(y_preds) == len(prob_preds) == len(texts)
        # return y_preds,prob_preds
        print(b_preds)

    

def main():
    MODEL_PATH = "./data/model_classify"
    MAX_LEN = 64
    BATCH_SIZE = 16
    NUM_LABEL = 3
    device = torch.device(0)
    # classifier = Classify(MODEL_PATH,device,MAX_LEN,BATCH_SIZE,NUM_LABEL)
    tokenizer =  AutoTokenizer.from_pretrained(MODEL_PATH, do_lower_case=True)

    model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels = NUM_LABEL,
            output_attentions = False,
            output_hidden_states = False,
        ).to(device)
    classifier = pipeline("sentiment-analysis",
                            model = model,
                            tokenizer = tokenizer,
                            device = 0,
                            framework = 'pt')
    
    preds = classifier('Việt Nam có bao nhiêu di sản văn hoá phi vật thể')
    print(preds)

if __name__ == "__main__":
    
    main()
    
