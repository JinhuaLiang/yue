## QuickStart

### 1. Prepare infer and tokenizer code

```bash
git clone https://github.com/multimodal-art-projection/YuE.git

cd YuE/inference/
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

### 2. Install environment and dependencies

```bash
module load apptainer
apptainer build --sandbox sandbox-yue sing-yue.def
```

### 3. Download pretrained weights

```bash
# Prepare the pretrained weight before running the container (Optional)
python3 setup.py
```

### 4. Run the inference

Now generate music with **YuE** using 🤗 Transformers. Make sure your step [1](#1-install-environment-and-dependencies) and [2](#2-download-the-infer-code-and-tokenizer) are properly set up.

Note:

- Set `--run_n_segments` to the number of lyric sections if you want to generate a full song. Additionally, you can increase `--stage2_batch_size` based on your available GPU memory.
- You may customize the prompt in `genre.txt` and `lyrics.txt`. See prompt engineering guide [here](#prompt-engineering-guide).
- You can increase `--stage2_batch_size` to speed up the inference, but be careful for OOM.
- LM ckpts will be automatically downloaded from huggingface.

```bash
# This is the CoT mode.
cd YuE/inference/
python3 infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1
```

We also support music in-context-learning (provide a reference song), there are 2 types: single-track (mix/vocal/instrumental) and dual-track.

Note:

- ICL requires a different ckpt, e.g. `m-a-p/YuE-s1-7B-anneal-en-icl`.
- Music ICL generally requires a 30s audio segment. The model will write new songs with similar style of the provided audio, and may improve musicality.
- Dual-track ICL works better in general, requiring both vocal and instrumental tracks.
- For single-track ICL, you can provide a mix, vocal, or instrumental track.
- You can separate the vocal and instrumental tracks using [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) or [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui).

```bash
# This is the dual-track ICL mode.
# To turn on dual-track mode, enable `--use_dual_tracks_prompt`
# and provide `--vocal_track_prompt_path`, `--instrumental_track_prompt_path`, 
# `--prompt_start_time`, and `--prompt_end_time`
# The ref audio is taken from GTZAN test set.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --use_dual_tracks_prompt \
    --vocal_track_prompt_path ../prompt_egs/pop.00001.Vocals.mp3 \
    --instrumental_track_prompt_path ../prompt_egs/pop.00001.Instrumental.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30 
```

```bash
# This is the single-track (mix/vocal/instrumental) ICL mode.
# To turn on single-track ICL, enable `--use_audio_prompt`, 
# and provide `--audio_prompt_path` , `--prompt_start_time`, and `--prompt_end_time`. 
# The ref audio is taken from GTZAN test set.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --use_audio_prompt \
    --audio_prompt_path ../prompt_egs/pop.00001.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30 
```

---

## Prompt Engineering Guide

The prompt consists of three parts: genre tags, lyrics, and ref audio.

### Genre Tagging Prompt

1. An example genre tagging prompt can be found [here](prompt_egs/genre.txt).
2. A stable tagging prompt usually consists of five components: genre, instrument, mood, gender, and timbre. All five should be included if possible, separated by space (space delimiter).
3. Although our tags have an open vocabulary, we have provided the top 200 most commonly used [tags](./top_200_tags.json). It is recommended to select tags from this list for more stable results.
4. The order of the tags is flexible. For example, a stable genre tagging prompt might look like: "inspiring female uplifting pop airy vocal electronic bright vocal vocal."
5. Additionally, we have introduced the "Mandarin" and "Cantonese" tags to distinguish between Mandarin and Cantonese, as their lyrics often share similarities.

### Lyrics Prompt

1. An example lyric prompt can be found [here](prompt_egs/lyrics.txt).
2. We support multiple languages, including but not limited to English, Mandarin Chinese, Cantonese, Japanese, and Korean. The default top language distribution during the annealing phase is revealed in [issue 12](https://github.com/multimodal-art-projection/YuE/issues/12#issuecomment-2620845772). A language ID on a specific annealing checkpoint indicates that we have adjusted the mixing ratio to enhance support for that language.
3. The lyrics prompt should be divided into sessions, with structure labels (e.g., [verse], [chorus], [bridge], [outro]) prepended. Each session should be separated by 2 newline character "\n\n".
4. **DONOT** put too many words in a single segment, since each session is around 30s (`--max_new_tokens 3000` by default).
5. We find that [intro] label is less stable, so we recommend starting with [verse] or [chorus].
6. For generating music with no vocal (instrumental only), see [issue 18](https://github.com/multimodal-art-projection/YuE/issues/18).

### Audio Prompt

1. Audio prompt is optional. Providing ref audio for ICL usually increase the good case rate, and result in less diversity since the generated token space is bounded by the ref audio. CoT only (no ref) will result in a more diverse output.
2. We find that dual-track ICL mode gives the best musicality and prompt following.
3. Use the chorus part of the music as prompt will result in better musicality.
4. Around 30s audio is recommended for ICL.
5. For music continuation, see [YuE-extend by Mozer](https://github.com/Mozer/YuE-extend). Also supports Colab.

---

## License Agreement \& Disclaimer

- The YuE model (including its weights) is now released under the **Apache License, Version 2.0**. We do not make any profit from this model, and we hope it can be used for the betterment of human creativity.
- **Use & Attribution**:
  - We encourage artists and content creators to freely incorporate outputs generated by YuE into their own works, including commercial projects.
  - We encourage attribution to the model’s name (“YuE by HKUST/M-A-P”), especially for public and commercial use.
- **Originality & Plagiarism**: It is the sole responsibility of creators to ensure that their works, derived from or inspired by YuE outputs, do not plagiarize or unlawfully reproduce existing material. We strongly urge users to perform their own due diligence to avoid copyright infringement or other legal violations.
- **Recommended Labeling**: When uploading works to streaming platforms or sharing them publicly, we **recommend** labeling them with terms such as: “AI-generated”, “YuE-generated", “AI-assisted” or “AI-auxiliated”. This helps maintain transparency about the creative process.
- **Disclaimer of Liability**:
  - We do not assume any responsibility for the misuse of this model, including (but not limited to) illegal, malicious, or unethical activities.
  - Users are solely responsible for any content generated using the YuE model and for any consequences arising from its use.
  - By using this model, you agree that you understand and comply with all applicable laws and regulations regarding your generated content.

---

## Acknowledgements

The project is co-lead by HKUST and M-A-P (alphabetic order). Also thanks moonshot.ai, bytedance, 01.ai, and geely for supporting the project.
A friendly link to HKUST Audio group's [huggingface space](https://huggingface.co/HKUSTAudio).

We deeply appreciate all the support we received along the way. Long live open-source AI!

---

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@misc{yuan2025yuescalingopenfoundation,
      title={YuE: Scaling Open Foundation Models for Long-Form Music Generation}, 
      author={Ruibin Yuan and Hanfeng Lin and Shuyue Guo and Ge Zhang and Jiahao Pan and Yongyi Zang and Haohe Liu and Yiming Liang and Wenye Ma and Xingjian Du and Xinrun Du and Zhen Ye and Tianyu Zheng and Yinghao Ma and Minghao Liu and Zeyue Tian and Ziya Zhou and Liumeng Xue and Xingwei Qu and Yizhi Li and Shangda Wu and Tianhao Shen and Ziyang Ma and Jun Zhan and Chunhui Wang and Yatian Wang and Xiaowei Chi and Xinyue Zhang and Zhenzhu Yang and Xiangzhou Wang and Shansong Liu and Lingrui Mei and Peng Li and Junjie Wang and Jianwei Yu and Guojian Pang and Xu Li and Zihao Wang and Xiaohuan Zhou and Lijun Yu and Emmanouil Benetos and Yong Chen and Chenghua Lin and Xie Chen and Gus Xia and Zhaoxiang Zhang and Chao Zhang and Wenhu Chen and Xinyu Zhou and Xipeng Qiu and Roger Dannenberg and Jiaheng Liu and Jian Yang and Wenhao Huang and Wei Xue and Xu Tan and Yike Guo},
      year={2025},
      eprint={2503.08638},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2503.08638}, 
}

@misc{yuan2025yue,
  title={YuE: Open Music Foundation Models for Full-Song Generation},
  author={Ruibin Yuan and Hanfeng Lin and Shawn Guo and Ge Zhang and Jiahao Pan and Yongyi Zang and Haohe Liu and Xingjian Du and Xeron Du and Zhen Ye and Tianyu Zheng and Yinghao Ma and Minghao Liu and Lijun Yu and Zeyue Tian and Ziya Zhou and Liumeng Xue and Xingwei Qu and Yizhi Li and Tianhao Shen and Ziyang Ma and Shangda Wu and Jun Zhan and Chunhui Wang and Yatian Wang and Xiaohuan Zhou and Xiaowei Chi and Xinyue Zhang and Zhenzhu Yang and Yiming Liang and Xiangzhou Wang and Shansong Liu and Lingrui Mei and Peng Li and Yong Chen and Chenghua Lin and Xie Chen and Gus Xia and Zhaoxiang Zhang and Chao Zhang and Wenhu Chen and Xinyu Zhou and Xipeng Qiu and Roger Dannenberg and Jiaheng Liu and Jian Yang and Stephen Huang and Wei Xue and Xu Tan and Yike Guo}, 
  howpublished={\url{https://github.com/multimodal-art-projection/YuE}},
  year={2025},
  note={GitHub repository}
}
```

<br>
