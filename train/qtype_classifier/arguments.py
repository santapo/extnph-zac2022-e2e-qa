import argparse

def load_args():
    #vinai/phobert-base
    #/home/ubuntu/embedding/bert4news/bert4news.pytorch
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pretrained', default='vinai/phobert-base')
    parser.add_argument('--model_save', default='./data/qtype_models/')
    parser.add_argument('--data_cache', default="./data/cache/")
    parser.add_argument('--data_path', default="./data/e2eqa-train+public_test-v1/classify_data.csv")
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--max_labels', default=3, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--epsilon', default=1e-8, type=float)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args
