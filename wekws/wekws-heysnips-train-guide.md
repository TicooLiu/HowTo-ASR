# wekws: wenet keyword spotting
> 项目地址:https://github.com/wenet-e2e/wekws.git

> 本篇内容为基于`hey snips`数据集格式进行自定义多唤醒词训练的操作步骤，均为本人实践记录。
> 项目源码中的示例仅演示了通过`hey snips`数据集进行单个唤醒词`hey snips`的训练,
> 但是实际项目中可能会出现多个唤醒词的情况，或者说是指令集的情况。因此对源码的训练脚本进行了
> 适当的修改，并增加了指令集文件，以实现多唤醒词识别的功能。

## 0. 环境准备
 - 架构：`12th Gen Intel(R) Core(TM) i9-12900H 2.50 GHz`
 - 显卡: `NVIDIA GeForce RTX 3070 Ti Laptop GPU`
 - 主机系统: `Windows 11 家庭中文版`
 - 训练系统：`Ubuntu 20.04(基于windows 11的WSL2.0)`

 **注意:** 以下操作，均为在训练系统中的操作。

### 0.1 下载miniconda软件安装脚本
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -P tools/
```
### 0.2 安装miniconda软件包
```
bash tools/Miniconda3-latest-Linux-x86_64.sh -b
```
### 0.3 初始化conda
```
$HOME/miniconda3/bin/conda init
```
### 0.4 创建python隔离环境
```
conda create -n wekws python=3.8
conda activate wekws
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch=1.10.0 torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

### 0.5 FFMPEG
在使用`bash`脚本划分自定义数据集时，需要获取每个音频文件的时长`duration`信息，此过程使用FFMPEG软件库功能，需要在`Ubuntu`环境下安装该软件库，如自己填写数据集文件，可不需此软件库。如自己写`python`脚本程序划分数据集，也可使用上述安装的`torchaudio`库，也不需安装FFMPEG。
```
sudo apt install ffmpeg
```

## 1 hey_snips数据集目录结构
> **注意:**
> 由于`hey snips`数据集无法公开下载，此处的数据集目录结构，以及划分数据集的`json`格式均是
> 参考`https://github.com/sonos/keyword-spotting-research-datasets.git`项目说明以及`wekws`示例源码得到。

 ### 1.1 数据集目录结构
 ```
 hey_snips_research_6k_en_train_eval_clean_ter/  
 |
 +-- audio_files/  ## 存放所有的音频文件
 |   |
 |   +-- <uuid>.wav
 |   |-- ...
 |   +-- <uuid>.wav
 |
 +-- dev.json    ## 应该是划分的验证数据集
 +-- test.json   ## 划分的测试数据集
 +-- train.json  ## 划分的训练数据集
 ```
 ### 1.2 划分数据集的json格式
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
 
 
## 2. 脚本流程
 > `wekws`源码`examples/hey_snips/s0/run.sh`。`run.sh`脚本的执行是分为多个独立的执行阶段来运行的，通过指定脚本执行参数可以控制脚本跨越的运行阶段，这样可以更好的独立调试每个运行阶段。也可以通过参数控制脚本执行所有的阶段(默认行为)。
 
 ### 脚本参数示例：
 - `bash run.sh --stage 0 --stop-stage 0`
   + 仅执行第0阶段的操作，执行完该阶段操作，立即结束。
 - `bash run.sh --stage 1 --stop-stage 4`
   + 执行第1阶段至第4阶段的操作，脚本顺序执行1，2，3，4四个阶段的操作，除非出现操作错误或人为中断，则脚本会执行到第4阶段的操作完成，然后立即结束。

 ### 脚本流程说明
  ```
  run.sh
  |  # --stage -1
  |  # 下载英文语音数据集hey_snips_kws_4.0.tar.gz
  |  # 解压hey_snips_kws_4.0.tar.gz至参数指定的目录:./data/local
  +-> local/snips_data_extract.sh --dl_dir $download_dir
  |
  |  # --stage 0
  |  # 准备数据
  |  # 1. 准备命令词文件
  |  # 2. 划分数据集，生成划分的json文件
  +-> local/prepare_data.py
  |   
  |
  |  # --stage 1
  |  # 计算CMVN并格式化数据集
  +-> tools/compute_cmvn_stats.py 
  +-> tools/wav_to_duration.sh
  +-> tools/make_list.py
  |
  |  # --stage 2
  |  # 开始训练
  +-> torchrun <参数>
  |   |
  |   +-> wekws/bin/train.py <参数>
  |
  |  # --stage 3
  |  # 模型平均化
  |  # 测试模型
  |  # 绘制检测曲线图
  +-> wekws/bin/average_model.py
  +-> wekws/bin/score.py
  +-> wekws/bin/plot_det_curve.py
  |
  |  # --stage 4
  |  # 导出两种格式的模型
  +-> wekws/bin/export_jit.py
  +-> wekws/bin/export_onnx.py
  ```

## 3. 自定义数据集训练`hey snips`模型
 ### 3.1 自定义数据集
 > 参照`1.1`节的目录结构组织存放自己的音频文件，此时自定义的数据集目录中还没有生成`dev.json`，`test.json`，`train.json`这三个文件。
 
 ### 3.2 划分并复制数据集
 #### 3.2.1 划分数据集
 > 将自定义数据集放在`~/hey_snips/`目录下，将脚本文件`hey-snips-split.sh`也放在该目录下，与自定义数据集同级，进入`~/hey_snips/`目录:`cd ~/hey_snips`，运行指令`bash hey-snips-split.sh <自定义数据目录>`，例如我的数据集目录名为`audio_data`，则运行指令为:`bash hey-snips-split.sh audio_data`。
 等待指令运行结束会在数据集同级目录生成`hey_snips_research_6k_en_train_eval_clean_ter`目录，该目录已经是标准的`hey_snips`格式的数据目录。

 **注意:** 由于扩展为多指令识别，在准备自定义数据集时，组织数据文件时借用了SpeechCommand的数据结构，目录结构如下:
 ```
 audio_data/
 |
 +-- <指令1>/
 |   |
 |   +-- <speaker_id>_nohash_<index>.wav
 |   ...
 |   +-- <speaker_id>_nohash_<index>.wav
 ...
 +-- <指令n>/
 |   |
 |   +-- <speaker_id>_nohash_<index>.wav
 |   ...
 |   +-- <speaker_id>_nohash_<index>.wav
 ```
在执行了划分数据集脚本文件后，生成的`json`文件中会比标准`hey snips`数据集的`json`文件多了额外的两个字段`word_id`和`word`，这两个字段的值是脚本遍历上述自定义的数据集目录自动生成的，在后续的操作中会用做生成命令词典。

 #### 3.2.2 复制数据集
 复制划分好的数据集目录`hey_snips_research_6k_en_train_eval_clean_ter`至`wekws/examples/hey_snips/s0/data/local/`目录下

 ### 3.3 自定义命令词文件
 > 自定义`command.txt`放在`./data/local/`下,
 > 与`hey_snips_research_6k_en_train_eval_clean_ter`
 > 同级，以供训练脚本使用。`command.txt`内容格式为每行一条命令词。

 **注意:** `command.txt`可用使用脚本遍历上述自定义的数据目录，截取`<指令>`部分自动生成，这样可以较好的保存与代码的一致性，避免出现指令变动时，数据表示不一致。


 ### 3.4 修改训练脚本
 > 1. 修改`wekws/example/hey_snips/s0/run.sh`,在`stage 0`中添加如下代码
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
 
 > 2. 修改`wekws/example/hey_snips/s0/local/prepare_data.py`,添加函数定义,并在`main`方法中调用
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
 > 修改main()函数中的调用
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
  
  
  
## 4 注意事项
  
  > 1. 如果训练时，内存不足，修改`conf/mdtc.yaml`中的`batch_size`
  > ```
  >     batch_conf:
  >       #batch_size: 100
  >       batch_size: 40
  > ```
  
  > 2. 训练时如出现`RuntimeError: Too many open files. Communication with the workers is no longer possible.`
  > 错误,执行下列指令修改打开的文件描述符上限,有些系统默认为1024.
  > ```
  > $ulimit -n 65535
  > ```
  > 或修改`run.sh`中`num_workers 8`为`num_workers 4`根据CPU线程数定义</br>
  
  > 3. 恢复训练
  > ```
  > $bash run.sh --stage 2 --stop_stage 2 --checkpoint exp/mdtc/7.pt
  > ```
  
  > 4. 查看训练过程
  > ```
  > $tensorboard --logdir tensorboard --port 12121 --bind_all
  > TensorBoard 2.13.0 at http://ubuntu:12121/ (Press CTRL+C to quit)
  > ```
  > 浏览器打开上述链接,进入TensorBoard页面查看
  
  > 5. 转换模型为移动设备可部署的模型
  > ```
  > $python -m onnxruntime.tools.convert_onnx_models_to_ort exp/mdtc/avg_10.onnx
  > ```
  
  > 6. 如使用onnxruntime进行验证，onnxruntime版本问题
  > ```
  > # 源码中默认使用的onnxruntime版本为1.12.0
  > # 修改runtime/onnxruntime/cmake/onnxruntime.cmake
  > set(ONNX_VERSION "1.15.0")
  > if(${ONNX_VERSION} MATCHES "1.15.0")
  >   set(URL_HASH "SHA256=1b3e88c0ea8e2770e5c9b11a36886097954a642c054b341992dcef268d8bb902")
  > elseif(${ONNX_VERSION} MATCHES "1.12.0")
  >   set(URL_HASH "SHA256=5d503ce8540358b59be26c675e42081be14a3e833a5301926f555451046929c5")
  > endif()
  > ```

## 5 结语
通过Android录音软件，根据自定义的`command.txt`约20条指令，收集了约4人的20轮次的无噪音朗读语音，训练出模型后，训练报告识别精度达到0.93，实际在Android设备部署测试，无噪音环境下，感觉基本识别精度达到0.7的样子，如大量增加训练数据，理论上在无噪音环境下应该可达到训练精度。仍需尝试给数据集加噪，测试随机噪音环境下的识别精度

**赞助**
> 如果有所收获，请微信意思意思

![image](https://github.com/TicooLiu/HowTo-ASR/blob/main/img/donate/weixin.jpeg)