import torch
import os
from model.candidates import *
from transformers import BertModel
from transformers import BertTokenizer
import torch.nn.functional as F
import numpy as np
from torch.utils.data import SequentialSampler, DataLoader
def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BertEncoder(torch.nn.Module):
    def __init__(self, bert_model,output_dim,add_linear=None):
        super(BertEncoder, self).__init__()
        bert_output_dim = bert_model.config.hidden_size

        self.bert_model = bert_model
        if add_linear:
            if output_dim==-1:
                output_dim = bert_model.config.hidden_size
            self.additional_linear = torch.nn.Linear(bert_output_dim, output_dim)
            self.dropout = torch.nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, input_ids, attention_mask, segment_ids):

        output = self.bert_model(
            input_ids, attention_mask, segment_ids)

        embeddings = output[0][:,0]

        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params.bert_model)
        cand_bert = BertModel.from_pretrained(params.bert_model)
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params.out_dim,
            add_linear=params.add_linear,
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params.out_dim,
            add_linear=params.add_linear,
        )
        self.config = ctxt_bert.config

    def forward(self, token_idx_ctxt, mask_ctxt, segment_idx_ctxt,
                token_idx_cands, mask_cands, segment_idx_cands,):

        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(token_idx_ctxt, mask_ctxt, segment_idx_ctxt)
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(token_idx_cands, mask_cands, segment_idx_cands)
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(params.bert_model,
                                                       do_lower_case=params.lowercase)
        # init model
        self.model = BiEncoderModule(self.params)

        self.model = self.model.to(self.device)
        self.data_parallel = params.data_parallel
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)


    def add_candidate_dataset(self,candidates_tensor,entity_freq=None):
        self.candidate_dataset = candidates_tensor
        self.entity_freq=None


    def get_candidates_encoding(self):
        ##每个epoch计算一次
        self.model.eval()
        sampler = SequentialSampler(self.candidate_dataset)
        dataloader = DataLoader(self.candidate_dataset, sampler=sampler, batch_size=self.params.eval_batch_size)
        encodings = []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            candidate_input, candidate_mask, candidate_segment = batch
            _, candidate_embeddings = self.model(None, None, None,
                                            candidate_input, candidate_mask, candidate_segment)

            encodings.append(candidate_embeddings)

        encodings = torch.cat(encodings, dim=0).data
        return encodings


    def update_tensor_with_candidate(self, data_tensor):
        self.model.eval()
        sampler = SequentialSampler(data_tensor)
        dataloader = DataLoader(data_tensor, sampler=sampler, batch_size=self.params.eval_batch_size)
        all_ids = []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            context_input, context_mask, context_segment, _, _, _, _, _ = batch
            context_embeddings, _ = self.model(context_input, context_mask, context_segment, None, None, None)
            # encodings.append(context_embeddings)
            scores, ids = self.Indexer.search_knn(context_embeddings.data.numpy(), self.params.top_k)  ##batch,k
            all_ids.append(ids)
        # encodings = torch.cat(encodings, dim=0)
        all_ids = np.concatenate(all_ids, 0)  # num_train, k
        # all_codings = candidate_tensor(torch.from_numpy(all_ids))
        # all_ids 最后应该是 #num_tain, k, embeddings

        data_tensor.add(torch.from_numpy(all_ids))
        return data_tensor


    def train_index(self):
        candidate_encoding = self.get_candidates_encoding()
        index_buffer = self.params.index_buffer
        vector_size = candidate_encoding.size(1)

        self.Indexer = DenseHNSWFlatIndexer(vector_size, index_buffer)

        self.Indexer.index_data(candidate_encoding.numpy())


    def score_candidate(self, context_ids, context_mask, context_segment,
                candidate_ids, candidate_mask, candidate_segment, hard_negs=False, cand_encs=None):


        embedding_ctxt, _ = self.model(context_ids, context_mask, context_segment, None, None, None)

        _, embedding_cands = self.model(None, None, None, candidate_ids, candidate_mask, candidate_segment) #batch,embedding
        if hard_negs:
            embedding_cands = torch.cat((torch.unsqueeze(embedding_cands,1),cand_encs),dim=1)

        if hard_negs:
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # num_mention_in_batch x 1 x embed_size
            embedding_cands = embedding_cands.permute(0,2,1) # num_mention_in_batch k=1， embed_size
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # num_mention_in_batch x 1 x k+1
            scores = torch.squeeze(scores)
            # (num_mention_in_batch,)
            return scores #batch,k+1
        else:
            all_scores = embedding_ctxt.mm(embedding_cands.t())
            return all_scores #batch,batch


    def forward(self, context_ids, context_mask, context_segment,
                candidate_input, candidate_mask, candidate_segment, label_idx,
                cands_id = None, in_batch_negs=True):

        hard_negs = not in_batch_negs

        if hard_negs:
            # 计算candidate encoding
            bs = len(cands_id)
            cands_id_ = torch.flatten(cands_id)
            cands_ids, cands_mask, cands_seg = self.candidate_dataset[cands_id_]
            _, cands_encoding = self.model(None,None,None,cands_ids,cands_mask,cands_seg)
            cands_encoding = torch.reshape(cands_encoding,[bs,self.params.top_k,-1])
        else:
            cands_encoding = None


        scores = self.score_candidate(context_ids, context_mask, context_segment,
                candidate_input, candidate_mask, candidate_segment, hard_negs=hard_negs, cand_encs=cands_encoding)

        bs = scores.size(0)

        if not hard_negs:
            target = torch.LongTensor(torch.arange(bs)) ##batch,batch
            target = target.to(self.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            # label_idx  batch
            # cans_id  batch,k
            label_idx = torch.unsqueeze(label_idx,-1)
            labels =  (torch.cat((label_idx,cands_id),dim=1) == label_idx.repeat(1,self.params.top_k+1)).float()
            loss_fct = torch.nn.BCEWithLogitsLoss(reduction="mean")
            loss = loss_fct(scores, labels)
        return loss, scores
