# YOLOv5 PT转Engine核心：RTX环境依赖配置全攻略（附避坑指南）

在实时目标检测场景里，YOLOv5训练生成的PT权重虽便捷，但推理速度远不能发挥RTX显卡的性能——而TensorRT优化后的Engine格式，能让推理效率提升30%-50%。网上充斥着PT转Engine的完整流程教程，但**没人系统讲过依赖配置的门道**。我在Ubuntu 24.04上反复踩坑十余次才打通流程，发现依赖版本匹配、安装顺序、源配置这三大块正是卡壳重灾区。这篇就聚焦最关键的依赖配置，帮你绕开90%的启动失败问题。

## 一、先厘清概念：别搞混TensorRTX与TensorRT

- **TensorRT**：NVIDIA官方推出的深度学习推理优化SDK，核心作用是对模型进行量化、剪枝等优化，生成高效的推理引擎（Engine文件），是“加速能力”的核心提供者。

- **TensorRTX**：wang-xinyu 开发的开源项目（[官方仓库](https://github.com/wang-xinyu/tensorrtx)），为YOLOv5等热门模型提供了适配TensorRT的代码实现，相当于“桥梁”——让我们能快速把YOLOv5的PT权重通过TensorRT转成Engine。

- **关键逻辑**：我们要配的依赖，本质是让“桥梁（TensorRTX）”能正常调用“加速核心（TensorRT）”，而这一切都依赖NVIDIA显卡的底层支持（驱动、CUDA、cuDNN）。

## 二、核心：RTX环境依赖配置全流程（附实测有效版本）

先放我最终打通的**版本匹配表**（重中之重！版本不匹配会报各种玄学错误，别乱换）：

|组件|实测有效版本|作用说明|
|---|---|---|
|操作系统|Ubuntu 24.04 LTS|稳定且对新显卡支持友好，建议用官方中国镜像源安装|
|NVIDIA显卡驱动|580系列（如580.xx.xx，选择非test、非server版本）|显卡底层驱动，需支持CUDA 12.0，选择非test、非server的稳定版本|
|CUDA|12.0（通过apt安装，对应cuda-12-0包）|显卡计算框架，TensorRT依赖其运行|
|cuDNN|8.9.2（for CUDA 12.0）|CUDA的深度学习加速库，提升TensorRT的卷积计算效率|
|TensorRT|8.6.1.6（TensorRT-8.6.1.6.Ubuntu-22.04.x86_64-gnu.cuda-12.0.tar.gz）|核心优化工具，负责生成Engine文件|
|Python环境|Miniconda 24.3.0 + Python 3.10|管理依赖包，避免系统环境污染|
|PyTorch|适配CUDA 12.0的版本（建议从PyTorch官网获取）|加载PT权重，配合TensorRTX执行转换|

### 2.1 第一步：配置系统源（避免后续安装卡壳）

Ubuntu默认源在国内速度慢，先换成清华源，同时配置conda和pip源，后续安装一路丝滑：

1. **系统源替换**：打开“软件和更新”，将“下载自”改为“位于中国的服务器”，刷新缓存后更新：
`sudo apt update && sudo apt upgrade -y`

2. **conda源配置**：安装Miniconda后，执行以下命令添加清华源：
        
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

        conda config --set show_channel_urls yes

3. **pip源配置**：创建pip配置文件：

        mkdir -p ~/.config/pip

        echo "[global] index-url = https://pypi.tuna.tsinghua.edu.cn/simple" > ~/.config/pip/pip.conf

### 2.2 第二步：安装NVIDIA显卡驱动（最容易忽略的前置）

驱动是基础，必须先装且版本适配CUDA！经过实测，Ubuntu自带的“附加驱动”能精准匹配系统内核，安装更便捷稳定，推荐优先使用此方式，重点选择“非test、非server”的最新稳定版本：

1. 卸载旧驱动（如果之前手动装过或有残留）：

        sudo apt purge nvidia-* -y

        # 执行后重启电脑

        sudo reboot

2. 打开“附加驱动”：通过Ubuntu搜索栏输入“附加驱动”打开工具，等待系统自动扫描适配的显卡驱动

3. 选择驱动版本：在扫描结果中，选择“非test（测试版）、非server（服务器版）”的最新稳定版本，例如我选择的“NVIDIA Corporation: NVIDIA GeForce RTX XXX”对应的580系列驱动

4. 应用驱动：点击“应用更改”，系统会自动下载并安装所选驱动，过程中需输入密码授权，等待安装完成（约5-10分钟，视网络速度而定）

5. 重启生效：安装完成后，点击“重启”按钮，或手动执行`sudo reboot`重启电脑

6. 验证：重启后打开终端，执行`nvidia-smi`，若能显示显卡型号、驱动版本（如580.xx.xx）及支持的CUDA版本（如“CUDA Version: 12.0”），则驱动安装成功

### 2.3 第三步：安装CUDA 12.0（严格匹配版本）

CUDA版本必须和TensorRT、cuDNN对应，这里选12.0：

1. 从[CUDA官网](https://developer.nvidia.com/cuda-12.0.1-download-archive)下载12.0版本（选择Linux→x86_64→Ubuntu→22.04→runfile）

2. 执行安装：严格按照官网步骤逐行代码执行即可。。。

3. 。。。

or 最简单安装：sudo apt install cuda

### 2.4 第四步：安装cuDNN 8.9.2（CUDA的“加速器”）

cuDNN是为深度学习优化的库，必须选对应CUDA 12.0的版本：

1. 从[cuDNN官网](https://developer.nvidia.com/rdp/cudnn-archive)下载“cuDNN Library for Linux x86_64”（版本8.9.7，for CUDA 12.x）

2. 执行安装：严格按照官网步骤逐行代码执行即可。。。

3. 。。。

### 2.5 第五步：安装TensorRT 8.6.1.6（核心转换工具）

TensorRT是生成Engine的关键，这里用tar包安装（比deb包更灵活）：

1. 从[TensorRT官网](https://developer.nvidia.com/nvidia-tensorrt-8x-download)下载对应版本（TensorRT-8.x）

2. 解压到指定目录：`tar -xzvf TensorRT-8.6.1.6.Ubuntu-22.04.x86_64-gnu.cuda-12.0.tar.gz -C ~/tools/`

3. 配置环境变量：打开`~/.bashrc`，添加：
        
        export TENSORRT_DIR=~/tools/TensorRT-8.6.1.6

        export PATH=$TENSORRT_DIR/bin:$PATH

        export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH

4. 生效环境变量：`source ~/.bashrc`

5. 安装Python绑定：进入TensorRT的python目录，根据当前Python版本选择对应whl文件安装（以Python 3.10为例）：`cd ~/tools/TensorRT-8.6.1.6/python && pip install tensorrt-8.6.1.6-cp310-none-linux_x86_64.whl`

6. 验证（可选）：Python中执行`import tensorrt as trt; print(trt.__version__)`，能输出8.6.1.6

### 2.6 第六步：配置Python环境（加载PT权重用）

用Miniconda创建独立环境，避免依赖冲突：

1. 创建环境：`conda create -n yolov5-trt python=3.10 -y`

2. 激活环境：`conda activate yolov5-trt`

3. **安装PyTorch**：建议前往[PyTorch官网](https://pytorch.org/get-started/locally/)，根据实际环境（CUDA 12.0、Python 3.10、pip）选择对应配置，复制官网提供的pip安装命令执行（官网命令会适配最新兼容版本，避免手动指定版本可能出现的适配问题）。

4. **安装YOLOv5依赖**：推荐通过YOLOv5官方的requirements.txt文件安装，确保依赖版本与YOLOv5适配。先克隆YOLOv5项目获取配置文件（若已克隆可跳过）：`git clone https://github.com/ultralytics/yolov5.git`，进入项目目录后执行：`pip install -r requirements.txt`（若需指定版本，可编辑requirements.txt后再安装）。

## 三、避坑指南：我踩过的5个致命错误（附解决方案）

这些问题网上搜不到明确答案，都是我实测踩坑总结的，遇到直接照解：

**坑1：通过apt安装CUDA时未清理旧驱动，导致冲突报错**

解决方案：apt安装CUDA时默认不会强制覆盖旧驱动，需先彻底清理残留：执行`sudo apt purge nvidia-* cuda-* -y`，重启后重新通过“附加驱动”装显卡驱动，再按步骤安装CUDA即可。

**坑2：Import tensorrt时报“libnvinfer.so.8: cannot open shared object file”**

解决方案：不是没装TensorRT，而是环境变量没生效！执行`source ~/.bashrc`重新加载，若还是报错，检查TENSORRT_DIR路径是否正确（确保和实际解压路径一致）。

**坑3：安装cuDNN后，执行TensorRT示例报错“CUDNN_STATUS_VERSION_MISMATCH”**

解决方案：cuDNN版本和CUDA不匹配！比如用了CUDA 12.0却装了for CUDA 11.8的cuDNN，重新下载对应版本的cuDNN（参考2.4步骤）。

**坑4：nvidia-smi显示CUDA Version 12.0，但nvcc -V显示11.7**

原因说明：
- `nvidia-smi`显示的是**显卡驱动支持的最高CUDA版本**（此处12.0代表驱动可兼容≤12.0的CUDA）；
- `nvcc -V`显示的是**当前系统实际生效的CUDA版本**（此处11.7说明环境变量指向了旧版本）。

两者不一致的核心问题是：系统中同时安装了多个CUDA版本（如11.7和12.0），但环境变量仍指向旧版本11.7，导致编译时调用的是旧版本。

解决方案：
1. 打开环境变量配置文件：`vim ~/.bashrc`（若不熟悉vim，可改用`gedit ~/.bashrc`图形化编辑）；
2. 找到与CUDA相关的路径设置（如`export PATH=/usr/local/cuda-11.7/bin:$PATH`和`export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH`）

- 删除旧版本（11.7）的配置，仅保留目标版本（12.0）的路径

- 例如：
`export PATH=/usr/local/cuda-12.0/bin:$PATH
` `export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH`
3. 保存文件后，执行`source ~/.bashrc`使配置生效，建议重启终端验证：
- 再次运行`nvcc -V`，确认显示版本为12.0；
- `nvidia-smi`显示的驱动支持版本无需修改，保持与实际安装的CUDA版本兼容即可（≤12.0）。

**坑5：Pip安装torch时速度极慢，甚至超时**

原因说明：
安装PyTorch（尤其是GPU版本）时速度慢的核心原因通常有两点：
1. 直接使用官方默认源（pypi.org）时，服务器位于境外，国内网络访问可能存在延迟或带宽限制；
2. 若配置了第三方镜像源（如清华源），可能因源地址错误、配置文件路径不正确（如非`~/.config/pip/pip.conf`或`~/.pip/pip.conf`），或镜像源同步延迟导致无法生效。

需注意：PyTorch官网提供的安装命令（含`https://download.pytorch.org/whl/`源）是针对GPU版本的官方推荐方式，该源本身无访问限制，但国内网络环境可能仍存在连接不稳定的问题。

解决方案：
1. **优先使用官网命令+国内镜像加速**：

- 例外：从PyTorch官网（https://pytorch.org/）获取对应CUDA 12.0版本的安装命令（如`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`），

2. **检查第三方镜像源配置**：

- 若需长期使用镜像源，确认`pip.conf`路径正确（Linux通常为`~/.config/pip/pip.conf`或`~/.pip/pip.conf`），文件内容示例：
`[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
` `trusted-host = pypi.tuna.tsinghua.edu.cn`
  保存后重新执行安装命令。

3. **临时解决网络问题**：

- 若上述方法仍超时，可尝试切换网络环境（如使用有线连接），或通过代理工具提升境外连接稳定性。

## 四、总结与后续：依赖配好，转换就成功了80%

很多人卡在PT转Engine的第一步，不是代码有问题，而是依赖配置没到位——毕竟NVIDIA的这套生态对版本匹配要求极高，差一个小版本都可能报错。我上面给的版本表和步骤都是实测跑通的，只要严格照做，基本能绕开所有基础坑。

配好依赖后，后续的PT转Engine核心步骤可参考官方教程（附录已提供权威指引）。如果执行中遇到其他问题，欢迎评论区留言，我会把你的问题补充到避坑指南里，帮助更多人少走弯路～

## 附录：PT转Engine核心流程指引

完成前文依赖配置后，PT转Engine的核心流程可直接参考TensorRTX官方提供的YOLOv5专属教程，该教程会根据YOLOv5版本动态更新适配步骤，比通用流程更精准权威：

官方教程链接：[https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)

官方教程核心优势：① 明确标注各YOLOv5版本（v5/v6/v7等）对应的分支切换方法；② 提供权重转换、编译推理的完整命令及常见问题解答；③ 实时更新适配最新系统环境的操作细节。

## 参考资料

- [1] TensorRTX官方仓库. [https://github.com/wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)

- [2] NVIDIA CUDA 12.0官方文档. [https://docs.nvidia.com/cuda/12.0/index.html](https://docs.nvidia.com/cuda/12.0/index.html)

- [3] YOLOv5配置与训练笔记. [https://www.cnblogs.com/tokepson/p/18817469](https://www.cnblogs.com/tokepson/p/18817469)

> （注：文档部分内容可能由 AI 生成，有错误请联系作者修改）