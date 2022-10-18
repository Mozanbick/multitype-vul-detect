## README

- **运行环境**

Python 3.9

Java 11

- **外部依赖软件**

Joern （官网最新版即可）

安装脚本：

```
$ wget https://github.com/joernio/joern/releases/latest/download/joern-install.sh
$ chmod +x ./joern-install.sh
$ sudo ./joern-install.sh
```

一般默认安装在`/home/<user>/bin/joern`下



#### 1. 处理源码

Java + CDT库过滤源文件保留C文件，并按照函数划分

代码在目录`<path_to_project>/joern/SGProcess/`下

建议把SARD数据集源码放在`<path_to_project>/joern/data/`下，假设这里路径是`<path_to_project>/joern/data/sard`，那么运行Java程序的命令如下：

```
$ cd ./joern/SCProcess/bin
$ java -cp ../bin:../lib/org/eclipse/cdt.core/5.6.0.201402142303/*:../lib/org/eclipse/equinox.common/3.6.200.v20130402-1505/* purge.ClassifyFiles ../../data/sard ../../data/raw_sard 1000
```

> 这里有三个参数：
>
> ../../data/sard 是源码文件路径
>
> ../../data/raw_sard 是保存路径
>
> 1000 是最大分组长度，因为把整个 SARD 放到 Joern 里面跑内存会不够
>
> 对于 SARD，1000的长度能够分成 8 组

如果出现无法运行的情况，那可能需要重新编译一下

```
$ cd ./joern/SCProcess
$ javac -cp lib/org/eclipse/cdt.core/5.6.0.201402142303/*:lib/org/eclipse/equinox.common/3.6.200.v20130402-1505/* -d bin src/purge/*.java
```

再运行最开始的程序即可。



#### 2. Joern

进入 Joern 的目录

```
$ cd ~/<user>/bin/joern/joern-cli
```

首先运行 joern-parse 脚本

```
$ joern-parse <path_to_project>/joern/data/raw_sard/group0 -o <path_to_project>/joern/joern-cli/parse/group0.bin
```

生成 group0 分组的代码数据库文件，对其他分组也是一样

然后运行 joern 脚本

```
$ ./joern
```

对于每一个数据库文件 .bin ，运行`extract-funcs-info.sc`以及`get-points.sc`脚本，其路径都在`<path_to_project>/joern/joern-cli/scripts`下，但是**注意运行之前请先修改脚本中的对应路径**

```
(joern)> importCpg("<path_to_project>/joern/joern-cli/parse/group0.bin")
(joern)> cpg.runScript("<path_to_project>/joern/joern-cli/scripts/extract-funcs-info.sc")
(joern)> cpg.runScript("<path_to_project>/joern/joern-cli/scripts/get-points.sc")
```

脚本运行的结果会保存在路径`<path_to_project>/joern/joern-cli/results/`下，这部分的内容就是对应代码的 CPG 信息。



#### 3. 运行Python脚本

进入项目主目录`<path_to_project>`

运行`process.py`脚本即可

**注意先修改脚本里的路径**

目前这个脚本的功能只是生成切片文件，切片文件保存在目录`<path_to_project>/joern/data/sequences`下。

