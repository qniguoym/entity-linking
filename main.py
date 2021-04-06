import time
from model.candidates import *
from utils.arguments import parser
from torch.utils.data import RandomSampler
from utils.utils import *
from model.model import *
from wiki_io import *


def main(args,logger):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    logger.info("STEP 1: Loading model.")
    reranker = BiEncoderRanker(args)
    tokenizer = reranker.tokenizer
    model = reranker.model
    device = reranker.device
    n_gpu = reranker.n_gpu
    # reranker = None
    # tokenizer = None
    # model = None
    # device = None
    # n_gpu = 1

    ## load entity2id Qid->id
    if not os.path.exists(args.entity2id):
        entity2freq = read_entity_to_count(args.entity_count_path) ##wikiepda得到
        logger.info('entity2freq {}'.format(len(entity2freq))) #60085608
        entity2def = read_title_to_id(args.entity_def_path) #wikidata得到 #12685964, 只有9种语言
        logger.info('entity2def {}'.format(len(entity2def)))
        tmp = sorted(entity2freq.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
        entity2id = {}
        for i,item in enumerate(tmp):
            e = item[0]
            if e in entity2def:
                e_id = entity2def[e]
                entity2id[e_id] = i
        with open(args.entity2id,'wb') as f:
            pickle.dump(entity2id,f)
    else:
        logger.info('Loading from {}'.format(args.entity2id))
        with open(args.entity2id,'rb') as f:
            entity2id = pickle.load(f)
    logger.info('Entity num is {}'.format(len(entity2id))) ## 7840236


    ##load_entity_description
    if not os.path.exists(args.entity2description):
        entity2descriptions = read_id_to_descr(args.entity_description_path) #70390762
        entity2description = get_description(entity2descriptions,args.prior_for_des_path, args.random_description)
        with open(args.entity2description,'wb') as f:
            pickle.dump(entity2description,f)

    else:
        logger.info('Loading from {}'.format(args.entity2description))
        with open(args.entity2description,'rb') as f:
            entity2description = pickle.load(f)
    logger.info('Entity description num is {}'.format(len(entity2description))) #70390762


    args.train_batch_size = args.train_batch_size//args.gradient_accumulation_steps
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    grad_acc_steps = args.gradient_accumulation_steps

    ###读数据
    train_data_path = os.path.join(args.tmp_path,'_'.join(args.lang)+'_'+str(args.random_description)+'.pkl')
    if not os.path.exists(train_data_path):
        train_samples = read_dataset('train', args.training_path, args.lang, debug=args.debug)
        valid_samples = read_dataset('dev', args.training_path, args.lang, debug=args.debug)
        logger.info("Read %d train samples." % len(train_samples))
        logger.info("Finished reading all train samples")
        logger.info("Read %d valid samples" % (len(valid_samples)))

        train_data, train_tensor_data = process_mention_data(
            samples=train_samples,
            tokenizer=tokenizer,
            entity2id=entity2id,
            entity2description=entity2description,
            max_context_length=args.max_context_length,
            max_cand_length=args.max_cand_length,
        )
        valid_data, valid_tensor_data = process_mention_data(
            valid_samples,
            tokenizer,
            entity2id=entity2id,
            entity2description=entity2description,
            max_context_length=args.max_context_length,
            max_cand_length=args.max_cand_length,
        )
        with open(train_data_path,'wb') as f:
            pickle.dump((train_data,train_tensor_data, valid_data,valid_tensor_data),f)

    else:
        with open(train_data_path,'rb') as f:
            train_data, train_tensor_data, valid_data, valid_tensor_data = pickle.load(f)


    logger.info("Read %d train tensor samples." % len(train_tensor_data))
    logger.info("Finished reading all train samples")


    ### 构建candidate data
    if not os.path.exists(args.candidates_data_path):
        candidates, candidates_tensor = get_candidates(entity2id,entity2description,tokenizer,
                                                       args.max_cand_length, args.debug)
        with open(args.candidates_data_path,'wb') as f:
            pickle.dump((candidates,candidates_tensor),f)
    else:
        with open(args.candidates_data_path,'rb') as f:
            candidates, candidates_tensor = pickle.load(f)

    reranker.add_candidate_dataset(candidates_tensor) ### 这个是永远保持不变的，并且与entity的id保持一致


    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size)

    write_to_file(os.path.join(args.output_path, "training_params.txt"), str(args))

    logger.info("Starting training")
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False))

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(args, optimizer, len(train_tensor_data), logger)

    model.train()
    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = args.epochs

    for epoch_idx in range(num_train_epochs):
        time_start = time.time()
        reranker.train()
        tr_loss = 0
        ### 由epoch决定是否进行hard_neg
        if epoch_idx>args.epoch_of_in_batch_neg:
            args.in_batch_neg = False
            ### 构建candidates encoding 和 index
        if not args.in_batch_neg:
            reranker.train_index()
            train_tensor_data = reranker.update_tensor_with_candidate(train_tensor_data)

        if args.shuffle:
            train_sampler = RandomSampler(train_tensor_data)
        else:
            train_sampler = SequentialSampler(train_tensor_data)

        train_dataloader = DataLoader(train_tensor_data, sampler=train_sampler, batch_size=train_batch_size)

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            context_ids, context_mask, context_segment, \
            candidate_input, candidate_mask, candidate_segment, \
            label_idx, candidates_id = batch

            loss, _ = reranker(context_ids, context_mask, context_segment,
                           candidate_input, candidate_mask, candidate_segment, label_idx, candidates_id, args.in_batch_neg)


            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (args.print_interval * grad_acc_steps) == 0:
                logger.info("Step {} - epoch {} average loss: {}\n".format(step,epoch_idx,
                                                                   tr_loss / (args.print_interval * grad_acc_steps)))
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (args.eval_interval * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                results = evaluate(reranker, valid_dataloader, args, device=device)
                logger.info('Step {} - epoch {} - results {}\n'.format(step,epoch_idx, results))
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(args.output_path, "epoch_{}".format(epoch_idx))
        save_model(model, tokenizer, epoch_output_folder_path)

        logger.info("Evaluation on the development dataset")
        results = evaluate(reranker, valid_dataloader, args, device=device)
        logger.info('Final epoch {} - results {}\n'.format(epoch_idx, results))

        ls = [best_score, results['1']]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info('Best score {} Best epoch {}\n'.format(best_score, best_epoch_idx))
        logger.info("\n")
        time_end = time.time()
        logger.info('Epoch {} Time {} \n'.format(epoch_idx,(time_end-time_start)/3600))
        exit()
    logger.info('Best score {} Best epoch {}'.format(best_score, best_epoch_idx))
    logger.info("\n")

if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    logger = get_logger(args.output_path)
    main(args,logger)
