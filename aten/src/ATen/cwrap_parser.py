import yaml

# follows similar logic to cwrap, ignores !inc, and just looks for [[]]



def parse(filename):
    # 打开文件并自动管理资源
    with open(filename, 'r') as file:
        declaration_lines = []  # 临时存储当前声明块的行内容
        declarations = []  # 存储所有解析后的声明对象
        in_declaration = False  # 标记是否处于声明块中
        
        # 逐行读取文件内容
        for line in file.readlines():
            line = line.rstrip()  # 移除行尾空白字符
            
            # 遇到声明块开始标记
            if line == '[[':
                declaration_lines = []  # 重置临时存储
                in_declaration = True  # 设置状态标记
                
            # 遇到声明块结束标记
            elif line == ']]':
                in_declaration = False
                # 将收集的行合并为字符串并用YAML解析
                declaration = yaml.load('\n'.join(declaration_lines))
                declarations.append(declaration)  # 存储解析结果
                
            # 当前行属于声明块内容
            elif in_declaration:
                declaration_lines.append(line)  # 收集声明内容行
                
        return declarations  # 返回所有解析后的声明对象
