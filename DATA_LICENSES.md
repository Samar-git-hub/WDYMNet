# Dataset and model licences

Nothing in this table is redistributed by this repository: `data/` is gitignored, and the
scripts download everything from the original sources. **Research use only.**

The one trained model we include, `results/checkpoints/stage_a_best.pt` (the 2.6 M-parameter fusion model), was
trained on MELD, MUStARD++ and the Helsinki Prosody Corpus. It contains no audio or text from them, but as a derived
work it inherits their research-only terms.

## Datasets

| Resource | Source | Licence / terms | Notes |
|---|---|---|---|
| MELD annotations | github.com/declare-lab/MELD (Poria et al., ACL 2019) | repository GPL-3.0; source material is *Friends* TV footage → research use only | CSVs `train/dev/test_sent_emo.csv` |
| MELD audio | huggingface.co/datasets/ajyy/MELD_audio | declared GPL-3.0 (same as MELD); 16 kHz mono FLAC extracted from the official MELD MP4s | used instead of the 10.9 GB official video tarball |
| MUStARD++ | github.com/cfiltnlp/MUStARD_Plus_Plus (Ray et al., LREC 2022) | released by the authors for research purposes (no explicit licence file in the repo); clips from TV sitcoms → research use only | CSV from GitHub; main-utterance videos from the authors' Google Drive folder |
| Helsinki Prosody Corpus | github.com/Helsinki-NLP/prosody (Talman et al., NoDaLiDa 2019) | data CC BY 4.0, code MIT | prominence labels |
| LibriTTS dev-clean / test-clean | openslr.org/60 (Zen et al., 2019) | CC BY 4.0 (derived from public-domain LibriVox) | downloaded from the OpenSLR EU mirror openslr.elda.org |

## Pretrained models (used frozen / partially fine-tuned, not redistributed)

| Model | Source | Licence |
|---|---|---|
| emotion2vec+ base | huggingface.co/emotion2vec/emotion2vec_plus_base (Ma et al., 2023) | code MIT; weights under the FunASR model licence (check the model card before any commercial use) |
| RoBERTa-base | huggingface.co/FacebookAI/roberta-base | MIT |
| Whisper small (faster-whisper / CTranslate2) | OpenAI Whisper, SYSTRAN/faster-whisper | MIT |
| MMS_FA forced aligner | torchaudio.pipelines.MMS_FA (Pratap et al., 2023) | CC BY-NC 4.0 (non-commercial) |
| Qwen3-4B-Instruct-2507 (rewriter), AWQ 4-bit build | huggingface.co/Qwen/Qwen3-4B-Instruct-2507, quantised build huggingface.co/Eslzzyl/Qwen3-4B-Instruct-2507-AWQ | Apache-2.0 |
| Silero VAD | github.com/snakers4/silero-vad | MIT |
