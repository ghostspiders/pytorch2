from optparse import OptionParser

# 创建命令行参数解析器
parser = OptionParser()
# 添加输出路径参数 -o/--output，默认值为当前目录
parser.add_option('-o', '--output', help='指定结果文件的输出路径',
                  action='store', default='.')
options, _ = parser.parse_args()

# 定义要处理的.cwrap文件列表（部分文件被注释掉了）
files = [
    # '../../csrc/cudnn/cuDNN.cwrap',
    '../../csrc/generic/TensorMethods.cwrap',
    # '../../csrc/generic/methods/SparseTensor.cwrap',
    '../../csrc/generic/methods/Tensor.cwrap',
    '../../csrc/generic/methods/TensorApply.cwrap',
    '../../csrc/generic/methods/TensorCompare.cwrap',
    '../../csrc/generic/methods/TensorCuda.cwrap',
    '../../csrc/generic/methods/TensorMath.cwrap',
    '../../csrc/generic/methods/TensorRandom.cwrap',
    # '../../csrc/generic/methods/TensorSerialization.cwrap',
]

# 存储所有声明行的列表
declaration_lines = []

# 遍历所有文件
for filename in files:
    with open(filename, 'r') as file:
        in_declaration = False  # 标记是否处于声明块中
        for line in file.readlines():
            line = line.rstrip()  # 去除行尾空白字符
            if line == '[[':  # 声明块开始
                in_declaration = True
                declaration_lines.append(line)
            elif line == ']]':  # 声明块结束
                in_declaration = False
                declaration_lines.append(line)
            elif in_declaration:  # 如果是声明块中的内容
                declaration_lines.append(line)

# 将所有声明行写入输出文件
with open(options.output, 'w') as output:
    output.write('\n'.join(declaration_lines) + '\n')
