# wekws: wenet keyword spotting
> Project: https://github.com/wenet-e2e/wekws.git

> This guide is train steps for `google speech command` dataset, these steps is record of my practice. On the basis, according to folder structure of `google speech command` dataset to prepare audio file, then to train custom command set.


## 0. Prepare environment
 - Architecture: `12th Gen Intel(R) Core(TM) i9-12900H 2.50 GHz`
 - Disply Adapter: `NVIDIA GeForce RTX 3070 Ti Laptop GPU`
 - Host System: `Windows 11 Home Edition`
 - Train System: `Ubuntu 20.04 based on WSL2.0 of windows 11`

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

## 1. Folder structure
 ### 1.1 speech_commands dataset
 Speech Command has two version，but the folder structure not change, just add or remove some audio data. Any can be used.

 ### 1.2 Dataset download link:
 - http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
 - http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

 ### 1.3 Folder structure after extracted
```
speech_commands_v0.02/  ## or speech_commands_v0.01
|
+--- <keyword>/    ## positive sample set
|    |
|    +--- <speaker-ID>_nohash_<index>.wav
|    +--- ...
|
+--- <non-keyword>/    ## negative sample set
|    +--- <speaker-ID>_nohash_<index>.wav
|    +--- ...
|
+--- _background_noise_/  ## background noise, can be used to data enhancement, 
|    |                    ## custom data may has no this folder，or copy official folder 
|    |                    ## into custom dataset.
|    +--- doing_the_dishes.wav
|    +--- dude_miaowing.wav
|    +--- exercise_bike.wav
|    +--- pink_noise.wav
|    +--- running_tap.wav
|    +--- white_noise.wav
|    +--- README.md
|
+--- LICENSE
+--- README.md
+--- testing_list.txt
+--- validation_list.txt
```

 
## 2. Flowchart of `run.sh`
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
  |  # download dataset and extract:
  |  # http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
  |  # split dataset to [train, valid, test]
  +-> local/data_download.sh  
  +-> local/split_dataset.py  
  |
  |  # --stage 0
  |  # prepare Kaldi format files
  +-> local/prepare_speech_command.py
  |   |
  |   +-> CLASSES specify keyword list,default begin from 'unknown'
  |   +-> folder name as keyword in CLASSES, match keyword index
  |   
  |
  |  # --stage 1
  |  # compute CMVN and format dataset
  +-> tools/compute_cmvn_stats.py 
  +-> tools/wav_to_duration.sh
  +-> tools/make_list.py
  |
  |  # --stage 2
  |  # start train
  +-> torchrun <parameters>
  |   |
  |   +-> wekws/bin/train.py <parameters>
  |
  |  # --stage 3
  |  # average model
  |  # test model
  +-> wekws/bin/average_model.py
  +-> wekws/bin/compute_accuracy.py
  |
  |  # --stage 4
  |  # export two format models
  +-> wekws/bin/export_jit.py
  +-> wekws/bin/export_onnx.py
  ```

## 3. Custom dataset to train Speech Command model
  ### 3.1 Split dataset
  After obtain audio data, saved int folder like `custom_dataset/`，put `speech-command-split.sh` with same level as data folder，then run it to 
  generate `testing_list.txt`和`validation_list.txt`，these two file format as：
  ```
  <keyword>/<speaker_id>_nohash_<index>.wav
  ...
  ```
  `<keyword>` is keyword, like `stop` etc. please arrange custom data folder as `1.3` description，can have no negative sampel set and noice sample set. Can do train only use postive sample set.

  > **Note:**
  - `speech-command-split.sh`is a bash script, is used to split dataset, coding it youself.
  - `custom_datase/` folder structure should uniform with `1.3` description。
  
  ### 3.2 Copy dataset
  - Copy all file in `custom_datase/` into `wekws/examples/speechcommand_v1/s0/data/local/speech_commands_v1/audio/`.
  - Copy generated `testing_list.txt` and `validation_list.txt` into `wekws/examples/speechcommand_v1/s0/data/local/speech_commands_v1/`, same level with `audio` folder.
  
  ### 3.3 Prepare keywords
  Prepare a file named `command.txt`, a keyword per linke. Copy `command.txt` into `wekws/examples/speechcommand_v1/s0/data/local/`，same level with `speech_commands_v1` folder.

### 3.4 Train model
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
  
  > 1. When trainning, occur out of memory issue, to change `batch_size` in `conf/mdtc.yaml`
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
  
  
  > 3. Recovery train
  > ```
  > $bash run.sh --stage 2 --stop_stage 2 --checkpoint exp/mdtc/7.pt
  > ```
  
  > 4. View progress of trainning
  > ```
  > $tensorboard --logdir tensorboard --port 12121 --bind_all
  > TensorBoard 2.13.0 at http://ubuntu:12121/ (Press CTRL+C to quit)
  > ```
  > Open above link with browser,view TensorBoard page
  
  > 5. Issue about export model:
  > After trainning done, when export model stage，`run.sh` default export two model：1. zip format model for jit 2. onnx format model. Export zip format model for jit is ok，but onnx format model error. The reason is `mdtc.yaml` has no `num_layers` argument, but python script used it. Committed issue to wekws，reply that correspondent argurment in `mdtc.yaml` is `num_stack`，need to modify `wekws/bin/export_onnx.py`, then re-export onnx model, [reference to wekws project #issue148](https://github.com/wenet-e2e/wekws/issues/148). To modify `main()` as below：
  >```python
  >  is_fsmn = configs['model']['backbone']['type'] == 'fsmn'
  >  is_mdtc = configs['model']['backbone']['type'] == 'mdtc'
  >  num_layers = configs['model']['backbone']['num_layers']
  >  ### github issue#148 said that the 'num_layers' should be 'num_stack'
  >  if is_mdtc 
  >      num_layers = configs['model']['backbone']['num_stack']
  >```

  > 6. Convert model for deploy to mobile device
  > ```
  > $python -m onnxruntime.tools.convert_onnx_models_to_ort exp/mdtc/avg_10.onnx
  > ```
  
  > 7. If verify model with onnxruntime，onnxruntime version issue
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
We obtain voice about 4 person with 20 loop per keyword via Android APP, and defined about 70 keywords. After finish trainning, we test model with test dataset get about `INFO Test Loss 0.11758032979251846 Acc 96.70860152135752`.
Because export onnx format model error, after modified `python` script，we can export onnx format model, but do not deply into Android device to do real test.

**DONATE**
> If feel helpful，please donate via WeChat, thanks!

![image](https://github.com/TicooLiu/HowTo-ASR/blob/main/img/donate/weixin.jpeg)