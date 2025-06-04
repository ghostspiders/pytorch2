import re

# 正则表达式：匹配模板中的 $identifier 或 ${identifier} 变量
# 支持以下特性：
# 1. 如果变量位于行首空白之后且值为列表，则进行块级替换（缩进并逐行显示）
# 2. 如果变量位于非空白行首且值为列表，则进行逗号分隔
# 3. 支持 ${,foo} 在非空列表前添加逗号
# 4. 支持 ${foo,} 在非空列表后添加逗号
substitution_str = r'(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})'

# 修复旧版Python的bug：\w* 无法正常工作
# 解决方案：替换为完整字符集 [a-zA-Z0-9_]*
# 参考：https://bugs.python.org/issue18647
substitution_str = substitution_str.replace(r'\w', r'[a-zA-Z0-9_]')

# 编译正则表达式，支持多行匹配
subtitution = re.compile(substitution_str, re.MULTILINE)

class CodeTemplate(object):
    """代码模板类，支持高级变量替换功能"""
    
    @staticmethod
    def from_file(filename):
        """从文件创建模板对象"""
        with open(filename, 'r') as f:
            return CodeTemplate(f.read())
    
    def __init__(self, pattern):
        """初始化模板对象"""
        self.pattern = pattern  # 原始模板内容
    
    def substitute(self, env={}, **kwargs):
        """执行模板替换
        参数:
            env: 包含替换变量的字典 (默认空字典)
            kwargs: 替换变量的关键字参数
        返回:
            替换后的字符串
        """
        # 变量查找函数：优先使用kwargs，其次使用env
        def lookup(v):
            return kwargs[v] if v in kwargs else env[v]
        
        # 缩进多行函数：为列表中的每个元素添加相同缩进
        def indent_lines(indent, v):
            lines = []
            for e in v:
                # 处理每个元素的多行内容
                for l in str(e).splitlines():
                    lines.append(indent + l)
            return "\n".join(lines) + "\n"
        
        # 替换处理函数
        def replace(match):
            indent = match.group(1)  # 捕获的缩进空格（可能为None）
            key = match.group(2)     # 捕获的变量名
            
            # 处理逗号修饰符
            comma_before = ''
            comma_after = ''
            if key.startswith("{"):
                key = key[1:-1]  # 移除花括号
                
                # 处理前置逗号：${,foo}
                if key.startswith(","):
                    comma_before = ', '
                    key = key[1:]
                
                # 处理后置逗号：${foo,}
                if key.endswith(","):
                    comma_after = ', '
                    key = key[:-1]
            
            # 获取变量值
            v = lookup(key)
            
            # 块级替换（变量在行首空白后且值为列表）
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]  # 转换为单元素列表
                return indent_lines(indent, v)
            
            # 行内列表替换（逗号分隔）
            elif isinstance(v, list):
                # 将列表元素连接为逗号分隔的字符串
                middle = ', '.join(str(x) for x in v)
                
                # 空列表直接返回
                if len(v) == 0:
                    return middle
                
                # 添加逗号修饰符
                return comma_before + middle + comma_after
            
            # 普通值替换
            else:
                return str(v)
        
        # 执行全局替换
        return self.subtitution.sub(replace, self.pattern)


# 测试代码
if __name__ == "__main__":
    # 创建模板实例
    c = CodeTemplate("""\
    int foo($args) {

        $bar
            $bar
        $a+$b
    }
    int commatest(int a${,stuff})
    int notest(int a${,empty,})
    """)
    
    # 执行替换并打印结果
    print(c.substitute(
        args=["hi", 8], 
        bar=["what", 7],
        a=3, 
        b=4, 
        stuff=["things...", "others"], 
        empty=[]
    ))

"""
预期输出:
    int foo(hi, 8) {

        what
        7
            what
            7
        3+4
    }
    int commatest(int a, things..., others)
    int notest(int a)
"""