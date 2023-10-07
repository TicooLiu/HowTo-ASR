# wekws: wenet keyword spotting
> 项目地址:https://github.com/wenet-e2e/wekws.git

> 本篇内容为基于`google speech command`数据集进行训练的操作步骤，均为本人实践记录。

## 0. 环境准备
 - 架构：`12th Gen Intel(R) Core(TM) i9-12900H 2.50 GHz`
 - 显卡: `NVIDIA GeForce RTX 3070 Ti Laptop GPU`
 - 主机系统: `Windows 11 家庭中文版`
 - 训练系统：`Ubuntu 20.04(基于windows 11的WSL2.0)`

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

## 1. 数据集目录结构
 ### 1.1 speech_commands数据集
 Speech Command数据集已经发布了两个版本，数据集目录结构没有变化，只是适当的增删了音频数据。选择哪个版本都可以使用。

 ### 1.2 数据集下载地址:
 - http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
 - http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

 ### 1.3 解压后数据集目录结构
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
+--- _background_noise_/    ## 背景噪音文件,可用于数据加噪,进行数据增强，自己的数据集可以没有该目录，也可直接复制官方数据集的此目录放入自己的数据集中
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

 
## 2. 脚本流程
 > `wekws`源码`examples/speechcommand_v1/s0/run.sh`。`run.sh`脚本的执行是分为多
 个独立的执行阶段来运行的，通过指定脚本执行参数可以控制脚本跨越的运行阶段，这样可以更好的
 独立调试每个运行阶段。也可以通过参数控制脚本执行所有的阶段(默认行为)。
 
 ### 2.1 脚本参数示例：
 - `bash run.sh --stage 0 --stop-stage 0`
   + 仅执行第0阶段的操作，执行完该阶段操作，立即结束。
 - `bash run.sh --stage 1 --stop-stage 4`
   + 执行第1阶段至第4阶段的操作，脚本顺序执行1，2，3，4四个阶段的操作，除非出现操作错误
  或人为中断，则脚本会执行到第4阶段的操作完成，然后立即结束。

 ### 2.2 脚本流程说明
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

## 3. 自定义数据集训练Speech Command模型
  ### 3.1 划分数据集
  在采集到的数据集`custom_dataset/`同级目录，运行`speech-command-split.sh`，
  为数据集生成`testing_list.txt`和`validation_list.txt`，这两个文本文件的格式为：
  ```
  <命令词>/<speaker_id>_nohash_<index>.wav
  ...
  ```
  其中`<命令词>`是自定义的命令词，可以是中文，例如：`你好`等。自定义的数据集目录组织需符合上述`1.3`节的目录结构，可以没有负样本数据目录以及噪音数据目录。仅使用正样本也可以进行模型训练。

  > **注意:**
  - `speech-command-split.sh`为自己编写的bash脚本，用于划分数据集，可自行编写。
  - `custom_datase/`目录的组织结构需符合上述的`1.3`节的目录结构一至。
  
  ### 3.2 复制数据集
  - 复制`custom_datase/`下的所有文件，至`wekws/examples/speechcommand_v1/s0/data/local/speech_commands_v1/audio/`目录下。
  - 复制生成的`testing_list.txt`和`validation_list.txt`文件至`wekws/examples/speechcommand_v1/s0/data/local/speech_commands_v1/`目录下，和`audio`目录同级。
  
  ### 3.3 准备命令词
  准备自定义的命令词文件`command.txt`，该文件内容为每行一条命令词。复制`command.txt`至`wekws/examples/speechcommand_v1/s0/data/local/`目录下，与`speech_commands_v1`目录同级。


  ## 4 注意事项
  
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
  
  > 5. 模型导出问题：
  > 训练完成后，在使用run.sh脚本导出模型时，run.sh脚本默认导出两种模型：1. jit的zip格式的模型文件 2. onnx格式的模型文件。导出jit的zip格式模型文件正常，导出onnx格式的模型文件时报告错误。其错误原因为mdtc.yaml的配置文件中没有num_layers参数，而python脚本中却使用了该参数。经过在项目issue中提问，回复说mdtc.yaml配置文件对应的参数为num_stack，需要修改python脚本`wekws/bin/export_onnx.py`，重新导出。[参见项目#issue148](https://github.com/wenet-e2e/wekws/issues/148)。导出脚本修改`main()`中的下列代码：
  >```python
  >  is_fsmn = configs['model']['backbone']['type'] == 'fsmn'
  >  is_mdtc = configs['model']['backbone']['type'] == 'mdtc'
  >  num_layers = configs['model']['backbone']['num_layers']
  >  ### github issue#148 said that the 'num_layers' should be 'num_stack'
  >  if is_mdtc 
  >      num_layers = configs['model']['backbone']['num_stack']
  >```

  > 6. 转换模型为移动设备可部署的模型
  > ```
  > $python -m onnxruntime.tools.convert_onnx_models_to_ort exp/mdtc/avg_10.onnx
  > ```
  
  > 7. onnxruntime版本问题
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
通过Android录音软件，根据自定义的`command.txt`约70条指令，收集了约4人的20轮次的无噪音朗读语音，训练出模型后，训练报告识别精度达到`INFO Test Loss 0.11758032979251846 Acc 96.70860152135752`。
由于在导出onnx模型文件时，出现错误，后经修改`python`脚本后，可正常导出onnx模型文件，但未实际部署测试。

**赞助**
> 如果有所收获，请微信意思意思

![image](https://github.com/TicooLiu/HowTo-ASR/blob/main/img/donate/weixin.jpeg)