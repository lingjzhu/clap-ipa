import pandas as pd
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.pre_tokenizers import Whitespace
import argparse
import sentencepiece as spm



def train_tokenizer(texts,vocab_size,outpath):
    
    tokenizer = Tokenizer(models.Unigram())
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size,special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    tokenizer.save(outpath)
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../phonemic_transcripts.txt')
    parser.add_argument('--outpath',type=str,default='../tokenizers/phone_unigram_tokenizer')
    parser.add_argument('--vocab_size',default=500,type=int)
    args = parser.parse_args()
    
    command = '--input=' + args.data + ' --model_type=unigram --model_prefix=' + args.outpath +' --vocab_size=' + str(args.vocab_size) + " --split_by_whitespace=true --character_coverage=1.0 --bos_id=1 --eos_id=2 --unk_id=3 --pad_id=0  --bos_piece=[CLS] --eos_piece=[SEP] --unk_piece=[UNK] --pad_piece=[PAD] --user_defined_symbols=[MASK] --byte_fallback=true"
    
    spm.SentencePieceTrainer.Train(command)
    
    # character
    #command = '--input=' + args.data + ' --model_type=bpe --model_prefix=' + args.outpath + " --split_by_whitespace=true --character_coverage=1.0 --bos_id=1 --eos_id=2 --unk_id=3 --pad_id=0  --bos_piece=[CLS] --eos_piece=[SEP] --unk_piece=[UNK] --pad_piece=[PAD] --user_defined_symbols=[MASK] --byte_fallback=true --use_all_vocab=true"