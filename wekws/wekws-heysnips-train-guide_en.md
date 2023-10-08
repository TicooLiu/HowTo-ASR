[中文](https://github.com/TicooLiu/HowTo-ASR/blob/main/wekws/wekws-heysnips-train-guide.md)

# wekws: wenet keyword spotting
> project on github:https://github.com/wenet-e2e/wekws.git

> This guide is step by step based on `hey snips` dataset to train costum 
> multi keywords. All steps are done by myself.
> Source code of `hey snips` example only is single keyword trainning demo. But 
> we may need multi keywords in actual project，or need command set。So modified 
> some source code and add command set text file，we can implement 
> multi keywords recognization function.Because I try it based on 
> `Speech Command` firstly，my audio files are also saved based on 
> `Speech Command` dataset folder. Then coding bash script to generate 
> json file according to requirement of `hey snips`dataset.

## 0. Prepare Environment
 - Architecture: `12th Gen Intel(R) Core(TM) i9-12900H 2.50 GHz`
 - Display Adapter:`NVIDIA GeForce RTX 3070 Ti Laptop GPU`
 - Host System: `Windows 11 Home Editon`
 - Train System: `Ubuntu 20.04(Based on WSL2.0 of windows 11)`

 **Note:** Below action is done in Train System

### 0.1 Download miniconda install script
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -P tools/
```
### 0.2 Install miniconda package
```
bash tools/Miniconda3-latest-Linux-x86_64.sh -b
```
### 0.3 Initial conda
```
$HOME/miniconda3/bin/conda init
```
### 0.4 Create python isolation environment
```
conda create -n wekws python=3.8
conda activate wekws
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch=1.10.0 torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

### 0.5 FFMPEG
When uses `bash` script to split costum dataset，we need get duration 
information of each audio file to generate json file, 
we use FFMPEG toolkit to do this.
If you fill duration information manually or coding python script with 
`torchaudio` to generate json file，you can skip this step.
```
sudo apt install ffmpeg
```

## 1 Folder structure of hey snips dataset
> **Note:**
> Because `hey snips` dataset can not open download，this structure 
> and json file of split dataset is reference to README
> `https://github.com/sonos/keyword-spotting-research-datasets.git` 
> and source code of `wekws` example.

 ### 1.1 Folder structure
 ```
 hey_snips_research_6k_en_train_eval_clean_ter/  
 |
 +-- audio_files/  ## all wav file saved in this folder
 |   |
 |   +-- <uuid>.wav
 |   |-- ...
 |   +-- <uuid>.wav
 |
 +-- dev.json    ## indicate which wav file is used to development
 +-- test.json   ## indicate which wav file is used to test
 +-- train.json  ## indicate which wav file is used to train
 ```
 ### 1.2 Format of json
 ```
 [
   { "id":"8c3ceef4-573e-4a79-8192-4ca6a4cba435", 
     "is_hotword":1,
	   "duration":0.880000,
	   "audio_file_path":"audio_files/8c3ceef4-573e-4a79-8192-4ca6a4cba435.wav"
   },
   ...
   { "id":"ade749f0-7e40-4e02-b091-4f87f18d9e20", 
     "is_hotword":1, 
	   "duration":0.840000, 
	   "audio_file_path":"audio_files/ade749f0-7e40-4e02-b091-4f87f18d9e20.wav" 
   }
 ]
 ```
 
 
## 2. Flowchart of script
 > `wekws` source code: `examples/hey_snips/s0/run.sh`.`run.sh` script is split to many 
 > stage to run, these stages can be run independently via parameters to control，This 
 > design is very good to test per stage independently. The script's default behavor is 
 > to run all stage.
 
 ### 2.1 Parameter of scritp example:
 - `bash run.sh --stage 0 --stop-stage 0`
   + Only do work of stage 0.
 - `bash run.sh --stage 1 --stop-stage 4`
   + Do work from stage 1 to 4 one by one.

 ### 2.2 Flowchart explaination
  ```
  run.sh
  |  # --stage -1
  |  # download dataset: hey_snips_kws_4.0.tar.gz
  |  # extract hey_snips_kws_4.0.tar.gz to folder:./data/local
  +-> local/snips_data_extract.sh --dl_dir $download_dir
  |
  |  # --stage 0
  |  # prepare data
  |  # 1. prepare keyword file
  |  # 2. split dataset, generate json file
  +-> local/prepare_data.py
  |   
  |
  |  # --stage 1
  |  # compute CMVN and format dataset
  +-> tools/compute_cmvn_stats.py 
  +-> tools/wav_to_duration.sh
  +-> tools/make_list.py
  |
  |  # --stage 2
  |  # do trainning
  +-> torchrun <parameters>
  |   |
  |   +-> wekws/bin/train.py <parameters>
  |
  |  # --stage 3
  |  # average model
  |  # test model
  |  # paint curve picutre
  +-> wekws/bin/average_model.py
  +-> wekws/bin/score.py
  +-> wekws/bin/plot_det_curve.py
  |
  |  # --stage 4
  |  # export two format models
  +-> wekws/bin/export_jit.py
  +-> wekws/bin/export_onnx.py
  ```

## 3. Custom dataset to train `hey snips` model
 ### 3.1 Custom dataset
 > Reference to `1.1` to save custom audio file，we do not have `dev.json`，`test.json`，`train.json` now.
 
 ### 3.2 Split and copy dataset
 #### 3.2.1 Split dataset
 > Put custom data into `~/hey_snips/`, and put `hey-snips-split.sh` into same folder，enter `~/hey_snips/` folder:`cd ~/hey_snips`, run command `bash hey-snips-split.sh <custom dataset folder>`, example my dataset folder is `audio_data`, then to run:`bash hey-snips-split.sh audio_data`.
 Waiting for finish, it will generate `hey_snips_research_6k_en_train_eval_clean_ter` folder，the folder is `hey_snips` format data folder.

 **Note:** because we want to recognize multiple keywords, we reference to folder structure of SpeechCommand dataset, as below:
 ```
 audio_data/
 |
 +-- <keyword 1>/
 |   |
 |   +-- <speaker_id>_nohash_<index>.wav
 |   ...
 |   +-- <speaker_id>_nohash_<index>.wav
 ...
 +-- <keyword n>/
 |   |
 |   +-- <speaker_id>_nohash_<index>.wav
 |   ...
 |   +-- <speaker_id>_nohash_<index>.wav
 ```
After run split script, generated `json` files have two more fields than normal `json` file of `hey snips` dataset:`word_id` and `word`，value of these two fields are generate via ergodic custom dataset folder that be used to make command dict in subsequent stage.

 #### 3.2.2 Copy dataset
 Copy `hey_snips_research_6k_en_train_eval_clean_ter` into `wekws/examples/hey_snips/s0/data/local/`

 ### 3.3 Custom keyword file
 > Put custom `command.txt` into `./data/local/`,
 > same level with `hey_snips_research_6k_en_train_eval_clean_ter`.
 > Content of `command.txt` is a keyword per line.

 **Note:** we can use script to traverse custom dataset folder, then cut out `<keyword>` to generate `command.txt`. This can keep uniformity when change keywords.


 ### 3.4 Modify trainning script
 > 1. Modify `wekws/example/hey_snips/s0/run.sh`, add below code in`stage 0`
 > ```bash
 > if [ -f $words ]; then
 >   rm -f $words
 > fi
 > echo "<filler> -1" > $words
 > #echo "Hey_Snips 0" >> dict/words.txt
 >
 > inx=0
 > while read -r line; do
 >   cmd=$(echo $line | sed 's/\r//g')
 >   echo "$cmd $inx" >> $words
 >   inx=$((inx + 1))
 > done < $download_dir/command.txt
 > ```
 
 > 2. Modify `wekws/example/hey_snips/s0/local/prepare_data.py`,add new function,and call it in `main`
 > ```python
 >def readWordMap(wordfile):
 >    wordmap = {}
 >    with open(wordfile, 'r') as wf:
 >        lines = wf.readlines()
 >    for line in lines:
 >        word,id = line.split()
 >        wordmap[word] = id
 >    return wordmap
 > ```
 > Modify function call in main()
 > ```python
 >words = readWordMap("./dict/words.txt")
 >with open(args.path, 'r', encoding='utf-8') as f:
 >    data = json.load(f)
 >    utt_id, label = [], []
 >    for entry in data:
 >        if entry['duration'] > 0:
 >            utt_id.append(entry['id'])
 >            #keyword_id = 0 if entry['is_hotword'] == 1 else -1
 >            keyword_id = words[entry['word']]
 >            label.append(keyword_id)
 > ```
  
### 3.5 Train model
> 1. Stage one by one
> ```
> # stage -1
> bash run.sh --stage -1 --stop-stage -1
> # stage 0
> bash run.sh --stage 0 --stop-stage 0
> # stage 1
> bash run.sh --stage 1 --stop-stage 1
> # stage 2
> bash run.sh --stage 2 --stop-stage 2
> # stage 3
> bash run.sh --stage 3 --stop-stage 3
> # stage 4
> bash run.sh --stage 4 --stop-stage 4
> ```

> 2. Stage all
> ```
> # stage all
> bash run.sh --stage -1 --stop-stage -4
> ```

> 3. Stage selection
> ```
> # stage 1 to 3
> bash run.sh --stage 1 --stop-stage 3
> ```
  
## 4 Matters needing attention
  
  > 1. When trainning，if out of memory，modify `batch_size` in `conf/mdtc.yaml`
  > ```
  >     batch_conf:
  >       #batch_size: 100
  >       batch_size: 40
  > ```
  
  > 2. When trainning, if occur `RuntimeError: Too many open files. Communication with the workers is no longer possible.`
  > error,run below command in bash to set file descriptor limit,
  > some syste default is 1024.
  > ```
  > $ulimit -n 65535
  > ```
  > or modify `num_workers 8` to `num_workers 4` in `run.sh` according to threads of CPU</br>
  
  > 3. Recovery trainning
  > ```
  > $bash run.sh --stage 2 --stop_stage 2 --checkpoint exp/mdtc/7.pt
  > ```
  
  > 4. View progress of trainning
  > ```
  > $tensorboard --logdir tensorboard --port 12121 --bind_all
  > TensorBoard 2.13.0 at http://ubuntu:12121/ (Press CTRL+C to quit)
  > ```
  > Open above link with browser,view TensorBoard page
  
  > 5. Convert model for deploy to mobile device
  > ```
  > $python -m onnxruntime.tools.convert_onnx_models_to_ort exp/mdtc/avg_10.onnx
  > ```
  
  > 6. If verify model with onnxruntime，onnxruntime version issue
  > ```
  > # default onnxruntime version is 1.12.0 in source code
  > # modify runtime/onnxruntime/cmake/onnxruntime.cmake to use other version
  > set(ONNX_VERSION "1.15.0")
  > if(${ONNX_VERSION} MATCHES "1.15.0")
  >   set(URL_HASH "SHA256=1b3e88c0ea8e2770e5c9b11a36886097954a642c054b341992dcef268d8bb902")
  > elseif(${ONNX_VERSION} MATCHES "1.12.0")
  >   set(URL_HASH "SHA256=5d503ce8540358b59be26c675e42081be14a3e833a5301926f555451046929c5")
  > endif()
  > ```

## 5 Concluding remarks
We obtain voice about 4 person with 20 loop per keyword via Android APP, and defined about 70 keywords. After finish trainning, we test model with test dataset get about Acc 0.93. We deploy into Android device to test in without noise environment, human feeling about Acc 0.7. If add more data and noise data to train, it maybe get more better.

**DONATE**
> If feel helpful，please donate via WeChat, thanks!

![image](https://github.com/TicooLiu/HowTo-ASR/blob/main/img/donate/weixin.jpeg)