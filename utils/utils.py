import numpy as np
import random
from tqdm import tqdm
import torch
import json
import os
import pickle
import sys
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, ConcatDataset
from transformers.optimization import get_linear_schedule_with_warmup
import logging
from transformers.optimization import AdamW
ENT_START_TAG = "[unused0]"
ENT_END_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"

def get_logger(output_dir=None):
    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    "{}/log.txt".format(output_dir), mode="a", delay=False
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger('entity_linking')
    logger.setLevel(10)
    return logger


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def read_dataset(mode, filename, lang, debug=False):
    samples = []
    for l in lang:
        txt_file_path = filename+'_'+l+'.jsonl'
        with open(txt_file_path, mode="r", encoding="utf-8") as file:
            for line in file:
                example = json.loads(line.strip())
                example['lang'] = l
                if mode == 'train' and not example['article_id'].endswith('3'):
                    samples.append(example)
                if mode == 'dev' and example['article_id'].endswith('3'):
                    samples.append(example)
                if debug and len(samples) > 200:
                    break

    return samples



def get_context_representation(sample,tokenizer,max_seq_length,ent_start_token=ENT_START_TAG,
                               ent_end_token=ENT_END_TAG,
):
    mention_tokens = []
    if sample['mention'] and len(sample['mention']) > 0:
        mention_tokens = tokenizer.tokenize(sample['mention'])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context = sample['context']
    title = sample['article_title']
    start = sample['start']
    end = sample['end']
    context_left = context[:start]
    context_right = context[end:]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)
    title_tokens = tokenizer.tokenize(title)

    #### title最多占1/4
    while len(title)>max_seq_length/4:
        title=title[:-1]

    max_context_length = max_seq_length/4*3
    left_quota = int((max_context_length - len(mention_tokens)) // 2 - 1) ###
    right_quota = int(max_context_length - len(mention_tokens) - left_quota - 3)
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    context_tokens = [tokenizer.cls_token] + title_tokens + [tokenizer.sep_token] + context_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    mask = [1] * len(input_ids) + [0] * (max_seq_length - len(input_ids))
    segment_id = [0]*(len(title_tokens)+2) + [1]*(max_seq_length-2-len(title_tokens))
    padding = tokenizer.convert_tokens_to_ids([tokenizer.pad_token]) * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
        'mask': mask,
        'segment':segment_id
    }


def process_mention_data(samples,tokenizer,entity2id, entity2description, max_context_length,max_cand_length,
                         ent_start_token=ENT_START_TAG,ent_end_token=ENT_END_TAG):
    processed_samples = []


    for idx, sample in enumerate(samples):
        if sample['entity'] not in entity2description:
            continue
        label_idx = entity2id[sample['entity']]
        description = entity2description[sample['entity']]

        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            ent_start_token,
            ent_end_token,
        )

        # label = sample[label_key]

        candi_tokens = get_candidate_representation(description, tokenizer, max_cand_length)
        record = {
            "context": context_tokens,
            'candidate':candi_tokens,
            'entity': sample['entity'],
            "entity_id": label_idx,
        }

        processed_samples.append(record)

    context_ids = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    context_mask = torch.tensor(
        select_field(processed_samples, "context", "mask"), dtype=torch.long,
    )
    context_segment = torch.tensor(
        select_field(processed_samples, "context", "segment"), dtype=torch.long,
    )

    cand_ids = torch.tensor(
        select_field(processed_samples, "candidate", "ids"), dtype=torch.long,
    )
    cand_mask = torch.tensor(
        select_field(processed_samples, "candidate", "mask"), dtype=torch.long,
    )
    cand_segment = torch.tensor(
        select_field(processed_samples, "candidate", "segment"), dtype=torch.long,
    )

    label_idx = torch.tensor(
        select_field(processed_samples, "entity_id"), dtype=torch.long,
    )
    data = {
        "context_ids": context_ids,
        'context_mask':context_mask,
        'context_segment':context_segment,
        "cand_ids": cand_ids,
        'cand_mask':cand_mask,
        'cand_segment':cand_segment,
        "label_idx": label_idx,
    }


    tensor_data = Dataset(context_ids, context_mask, context_segment, cand_ids, cand_mask, cand_segment, label_idx)

    return data, tensor_data


def get_candidate_representation(candidate_desc, tokenizer,max_seq_length):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)


    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    mask = [1] * len(input_ids) + [0] * (max_seq_length - len(input_ids))
    segment_id = [0] * max_seq_length
    padding = tokenizer.convert_tokens_to_ids([tokenizer.pad_token]) * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
        'mask':mask,
        'segment':segment_id
    }


def load_candidates(entity_catalogue, entity_encoding, faiss_index=None,
                    index_path=None, logger=None):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        # candidate_encoding = torch.load(entity_encoding)
        indexer = None
    else:
        if logger:
            logger.info("Using faiss index to retrieve entities.")
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        # if faiss_index == "flat":
        #     indexer = DenseFlatIndexer(1)
        # elif faiss_index == "hnsw":
        #     indexer = DenseHNSWFlatIndexer(1)
        # else:
        #     raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        # indexer.deserialize_from(index_path)

    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            print(entity)
            exit()
            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
    )


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)


def get_optimizer(model, args):
    return get_bert_optimizer(
        [model],
        args.type_optimization,
        args.learning_rate,
        args.fp16,
        args
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params.train_batch_size
    epochs = params.epochs

    num_train_steps = int(len_train_data/batch_size) * epochs
    num_warmup_steps = int(num_train_steps * params.warmup_proportion)

    # (optimizer, num_warmup_steps, num_training_steps, last_epoch=-1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def get_bert_optimizer(models, type_optimization, learning_rate, fp16, args):
    """ Optimizes the network with AdamWithDecay
    """
    patterns_optimizer = {
        'additional_layers': ['additional'],
        'top_layer': ['additional', 'bert_model.encoder.layer.11.'],
        'top4_layers': [
            'additional',
            'bert_model.encoder.layer.11.',
            'encoder.layer.10.',
            'encoder.layer.9.',
            'encoder.layer.8',
        ],
        'all_encoder_layers': ['additional', 'bert_model.encoder.layer'],
        'all': ['additional', 'bert_model.encoder.layer', 'bert_model.embeddings'],
    }

    if type_optimization not in patterns_optimizer:
        print(
            'Error. Type optimizer must be one of %s' % (str(patterns_optimizer.keys()))
        )
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    no_decay = ['bias', 'gamma', 'beta']
    patterns = patterns_optimizer[type_optimization]

    for model in models:
        for n, p in model.named_parameters():
            if any(t in n for t in patterns):
                if any(t in n for t in no_decay):
                    parameters_without_decay.append(p)
                    parameters_without_decay_names.append(n)
                else:
                    parameters_with_decay.append(p)
                    parameters_with_decay_names.append(n)

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay': args.weight_decay},
        {'params': parameters_without_decay, 'weight_decay': 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        correct_bias=False
    )

    # if fp16:
    #     optimizer = fp16_optimizer_wrapper(optimizer)

    return optimizer


def save_model(model, tokenizer, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def get_description(entity2descriptions,prior_path,random_description):
    ###data-driven selection
    entity2description = {}
    if random_description:
        for entity, lang_list in entity2descriptions.items():
            if 'en' in lang_list:
                entity2description[entity] = lang_list['en']
            else:
                line = random.choice(list(lang_list.values()))
                entity2description[entity] = line
    else:
        with open(prior_path,'rb') as f:
            entitylangnum, langnum = pickle.load(f)
        for entity, lang_list in entity2descriptions.items():
            if entity not in entitylangnum:
                continue
            for key,value in entitylangnum[entity].items():
                entitylangnum[entity][key] = [value, langnum[key]]
            cur_lang_num = sorted(entitylangnum[entity].items(),key=lambda x:(x[1][0],-x[1][1]),reverse=True)

            for item in cur_lang_num:
                l = item[0]
                if l in lang_list:
                    entity2description[entity]=lang_list[l]
                    break
    return entity2description

class Dataset(torch.utils.data.Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.candidates=None

    def add(self, candidates_tensor):
        self.candidates = candidates_tensor

    def __getitem__(self, index):
        if self.candidates is not None:
            cand = self.candidates[index]
        else:
            cand = self.tensors[0][index] ##没用

        batch = [tensor[index] for tensor in self.tensors]
        batch.append(cand)
        return tuple(batch)

    def __len__(self):
        return self.tensors[0].size(0)

def get_data(data_tensor, candidate_tensor, indexer,model, device, args):
    '''
    :param data: {context: entity: entity_id}
    :param indexer:
    :param candidates:
    :return:
    '''
    model.eval()
    sampler = SequentialSampler(data_tensor)
    dataloader = DataLoader(data_tensor, sampler=sampler, batch_size=args.eval_batch_size)
    all_ids = []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        context_input, context_mask, context_segment,_,_,_,_,_= batch
        context_embeddings, _ = model(context_input, context_mask, context_segment,None,None,None)
        # encodings.append(context_embeddings)
        scores, ids = indexer.search_knn(context_embeddings.data.numpy(), args.top_k) ##batch,k
        all_ids.append(ids)
    # encodings = torch.cat(encodings, dim=0)
    all_ids = np.concatenate(all_ids,0) #num_train, k
    # all_codings = candidate_tensor(torch.from_numpy(all_ids))
    # all_ids 最后应该是 #num_tain, k, embeddings

    data_tensor.add(torch.from_numpy(all_ids))

    return data_tensor



def evaluate(reranker, eval_dataloader, params, device):
    reranker.model.eval()

    nb_eval_examples = 0
    ###第一步 更新candidates encoding
    reranker.train_index()
    ### 计算recall 1和100
    max_k = max(params.eval_list)
    all_ids = []
    labels = []
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        context_ids, context_mask, context_segment, \
        candidate_input, candidate_mask, candidate_segment, label_idx, _ = batch
        with torch.no_grad():
            context_embeddings, _ = reranker.model(context_ids, context_mask, context_segment, None, None, None)
            scores, ids = reranker.Indexer.search_knn(context_embeddings.data.numpy(), max_k)  ##batch,k
            all_ids.append(ids)
            labels.append(label_idx.data.numpy())
            nb_eval_examples += context_ids.size(0)

    labels = np.concatenate(labels,axis=0) #n
    all_ids = np.concatenate(all_ids,axis=0) #n,k
    base_recall = {str(topk): 0 for topk in params.eval_list}
    for label, ids in zip(labels,all_ids):
        for j, k in enumerate(params.eval_list):
            if label in ids[:k]:
                base_recall[str(k)]+=1

    for key in base_recall.keys():
        base_recall[key] = base_recall[key]/nb_eval_examples

    return base_recall










