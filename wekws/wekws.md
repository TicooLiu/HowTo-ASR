# wekws: wenet keyword spotting
> 项目地址:https://github.com/wenet-e2e/wekws.git

## 0. 环境
```
Ubuntu 20.04
# download the miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -P tools/
# install the miniconda
bash tools/Miniconda3-latest-Linux-x86_64.sh -b
# conda init
$HOME/miniconda3/bin/conda init

conda create -n wekws python=3.8
conda activate wekws
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch=1.10.0 torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

## 1. 数据集目录结构
 ### 1.1 speech_commands数据集
```
speech_commands_v0.02/  ## 或者speech_commands_v0.01
|
+--- <命令词>/    ## 正样本数据
|    |
|    +--- <发音者ID>_nohash_<第n遍>.wav
|    +--- ...
|
+--- <非命令词>/    ## 负样本数据
|    +--- <发音者ID>_nohash_<第n遍>.wav
|    +--- ...
|
+--- _background_noise_/    ## 背景噪音文件,可用于数据加噪,进行数据增强
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

 ### 1.2 hey_snips数据集
 #### 1.2.1 数据集目录结构
 ```
 hey_snips_research_6k_en_train_eval_clean_ter/  
 |
 +-- audio_files/  ## 存放所有的音频文件
 |   |
 |   +-- <uuid>.wav
 |
 +-- dev.json    ## 应该是划分的验证数据集
 +-- test.json   ## 划分的测试数据集
 +-- train.json  ## 划分的训练数据集
 ```
 #### 1.2.2 划分数据集的json格式
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
 
 #### 1.2.3 自定义命令词文件
 > 自定义`command.txt`放在`/data/local/`下,
 > 与`hey_snips_research_6k_en_train_eval_clean_ter`
 > 同级，以供训练脚本使用。`command.txt`内容格式为每行一条命令词。

 ### 1.2.4 修改训练脚本
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
 
## 2. 脚本流程
 ### 2.1 speechcommand:`examples/speechcommand_v1/s0`
  ```
  run.sh
  |  # --stage -1
  |  # 下载英文语音数据集并解压:http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
  |  # 划分数据集[train, valid, test]
  +-> local/data_download.sh  
  +-> local/split_dataset.py  
  |
  |  # --stage 0
  |  # 准备Kaldi格式的文件
  +-> local/prepare_speech_command.py
  |   |
  |   +-> CLASSES中指定要识别的命令词列表,默认从`unknown`开始
  |   +-> 根据数据集目录结构匹配命令词,数据集中命令词目录名在CLASSES中,则匹配相应的index
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
  +-> wekws/bin/average_model.py
  +-> wekws/bin/compute_accuracy.py
  |
  |  # --stage 4
  |  # 导出模型
  +-> wekws/bin/export_jit.py
  +-> wekws/bin/export_onnx.py
  ```
 ### 2.2 hey_snips:`examples/hey_snips/s0`

## 3. 自定义数据集训练speechcommand模型
  ### 3.1 划分数据集
  在采集到的数据集`sgm_speech/`同级目录，运行`speech-command-split.sh`，
  为数据集生成`testing_list.txt`和`validation_list.txt`
  
  ### 3.2 复制数据集
  复制划分好的数据集文件至`wekws/examples/speechcommand_v1/s0/data/local/speech_commands_v1/audio/`目录下
  
  ### 3.3 准备命令词
  
  > ## **注意:**
  
  > 0. 准备command.txt文件，每行为一条命令，放在data/local/目录下
  
  > 1. 如果训练时，内存不足，修改`conf/mdtc.yaml`中的`batch_size`
  > ```
  >     batch_conf:
  >       #batch_size: 100
  >       batch_size: 40
  > ```
  
  > 2. 训练时如出现`RuntimeError: Too many open files. Communication with the workers is no longer possible.`
  > 错误,执行下列指令修改打开的文件描述符上限,系统默认为1024.
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
  
  > 6. onnxruntime版本问题
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

## 4. 自定义数据集训练hey_snips模型