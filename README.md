# CLAP-IPA
Paper: [The taste of IPA: Towards open-vocabulary keyword spotting and forced alignment in any language](https://arxiv.org/abs/2311.08323) To appear in NAACL 2024.   
   
We are gradually releasing the data and code. Thank you for your patience.

### Usage


### Pretrained Models

| Model | Phone Encoder | Speech encoder |
|---|---|---|
| CLAP-IPA-tiny | `anyspeech/clap-ipa-tiny-phone` | `anyspeech/clap-ipa-tiny-speech` |
| CLAP-IPA-base | `anyspeech/clap-ipa-base-phone` | `anyspeech/clap-ipa-base-speech` |
| CLAP-IPA-small | `anyspeech/clap-ipa-small-phone` | `anyspeech/clap-ipa-small-speech` |
| IPA-Aligner-tiny | `anyspeech/ipa-align-tiny-phone` | `anyspeech/ipa-align-tiny-speech` |
| IPA-Aligner-base | `anyspeech/ipa-align-base-phone` | `anyspeech/ipa-align-base-speech` |
| IPA-Aligner-small | `anyspeech/ipa-align-small-phone` | `anyspeech/ipa-align-base-speech` |


### IPA Pack
All datasets are distributed as `wds` files on huggingface hub.   
 - **FLEURS-IPA**: https://huggingface.co/datasets/anyspeech/fleurs_ipa
 - **MSWC-IPA**: https://huggingface.co/datasets/anyspeech/mswc_ipa
 - **DORECO-IPA**: https://huggingface.co/datasets/anyspeech/doreco_ipa

After this study, we found that these datasets still contain inconsistent unicode encoding of IPA symbols.  
**A cleaner version will be released when we finish another round of data cleaning**.

#### To download these datasets:
```
from huggingface_hub import snapshot_download

snapshot_download(repo_id="anyspeech/fleurs_ipa", repo_type="dataset", local_dir="your_own_folder",local_dir_use_symlinks=False,resume_download=False,max_workers=4)

```

#### To load webdataset files:
```
import webdataset as wds  # Note the typical import shorthand
dataset = (
      wds.WebDataset("data-archives/shard-00{00...24}.tar")  # 25 shards
      .decode()  # Automagically decode files
      .shuffle(size=1000)  # Shuffle on-the-fly in a buffer
      .batch(batchsize=10)  # Create batches
)
```
