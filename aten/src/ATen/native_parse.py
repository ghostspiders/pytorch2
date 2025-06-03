from __future__ import print_function  # 确保Python 2/3兼容性
import re  # 正则表达式模块
import yaml  # YAML解析库
import pprint  # 美化打印
import sys  # 系统相关功能

try:
    # 优先使用C加速的YAML加载器
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # 回退到纯Python实现


def parse_default(s):
    """解析默认值字符串为Python对象"""
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    elif s == 'nullptr':
        return s  # C++空指针保留原样
    elif s == '{}':
        return '{}'  # 空字典保留原样
    elif re.match(r'{.*}', s):
        return s  # 字典/集合字面量保留原样
    elif s == 'None':
        return 'c10::nullopt'  # Python None转换为C++可选类型
    try:
        return int(s)  # 尝试解析为整数
    except Exception:
        try:
            return float(s)  # 尝试解析为浮点数
        except Exception:
            return s  # 其他情况保留字符串


def sanitize_type(typ):
    """规范化类型字符串"""
    if typ == 'Generator*':
        return 'Generator *'  # 确保生成器指针格式一致
    return typ


def sanitize_types(types):
    """处理可能包含元组的类型声明"""
    if types[0] == '(' and types[-1] == ')':
        # 将元组类型拆分为单独类型列表
        return [sanitize_type(x.strip()) for x in types[1:-1].split(',')]
    return [sanitize_type(types)]  # 单个类型包装为列表


def parse_arguments(args, func_decl, func_name, func_return):
    """解析函数参数定义"""
    arguments = []
    python_default_inits = func_decl.get('python_default_init', {})
    is_out_fn = func_name.endswith('_out')  # 判断是否为输出函数
    
    # 输出函数必须明确声明variant为function
    if is_out_fn and func_decl.get('variants', []) not in [[], 'function', ['function']]:
        raise RuntimeError("Native functions suffixed with _out MUST be declared with only the function variant; "
                           "e.g., variants: function; otherwise you will tickle a Python argument binding bug "
                           "(which usually manifests itself as the result variable being undefined.) "
                           "The culprit was: {}".format(func_name))
    kwarg_only = False  # 标记是否进入仅关键字参数部分

    if len(args.strip()) == 0:
        return arguments  # 无参数直接返回

    # 简单参数解析(注意: 复杂模板类型可能解析错误)
    for arg_idx, arg in enumerate(args.split(', ')):
        type_and_name = [a.strip() for a in arg.rsplit(' ', 1)]  # 从右边分割类型和名称
        if type_and_name == ['*']:
            assert not kwarg_only
            kwarg_only = True  # *标记之后为仅关键字参数
            continue

        t, name = type_and_name
        default = None
        python_default_init = None

        if '=' in name:
            ns = name.split('=', 1)  # 分离默认值
            name, default = ns[0], parse_default(ns[1])

        if name in python_default_inits:
            assert default is None
            python_default_init = python_default_inits[name]  # 获取Python特有默认值

        typ = sanitize_types(t)
        assert len(typ) == 1
        argument_dict = {
            'type': typ[0].rstrip('?'),  # 移除可空标记
            'name': name, 
            'is_nullable': typ[0].endswith('?')  # 标记可空类型
        }
        
        # 处理固定大小的IntList
        match = re.match(r'IntList\[(\d+)\]', argument_dict['type'])
        if match:
            argument_dict['type'] = 'IntList'
            argument_dict['size'] = int(match.group(1))
            
        if default is not None:
            argument_dict['default'] = default
        if python_default_init is not None:
            argument_dict['python_default_init'] = python_default_init
            
        # 输出函数的前几个参数对应返回值
        if is_out_fn and arg_idx < len(func_return):
            argument_dict['output'] = True
            
        if kwarg_only:
            argument_dict['kwarg_only'] = True  # 标记仅关键字参数

        arguments.append(argument_dict)
    return arguments


def parse_return_arguments(return_decl, inplace):
    """解析返回值定义"""
    arguments = []
    # 处理可能的多返回值(元组形式)
    if return_decl[0] == '(' and return_decl[-1] == ')':
        return_decl = return_decl[1:-1]
    multiple_args = len(return_decl.split(', ')) > 1

    for arg_idx, arg in enumerate(return_decl.split(', ')):
        type_and_maybe_name = [a.strip() for a in arg.rsplit(' ', 1)]
        if len(type_and_maybe_name) == 1:
            t = type_and_maybe_name[0]
            # 就地操作返回self，否则使用result/resultN命名
            name = 'self' if inplace else 'result' if not multiple_args else 'result' + str(arg_idx)
        else:
            t, name = type_and_maybe_name

        typ = sanitize_type(t)
        argument_dict = {'type': typ, 'name': name, 'output': True}
        arguments.append(argument_dict)
    return arguments


def has_sparse_dispatches(dispatches):
    """检查是否有稀疏张量调度"""
    for dispatch in dispatches:
        if 'Sparse' in dispatch:
            return True
    return False


def parse_native_yaml(path):
    """解析YAML格式的native函数定义文件"""
    with open(path, 'r') as f:
        return yaml.load(f, Loader=Loader)


def run(paths):
    """主处理函数，解析所有YAML文件"""
    declarations = []
    for path in paths:
        for func in parse_native_yaml(path):
            declaration = {'mode': 'native'}
            try:
                # 分割函数声明和返回类型
                if '->' in func['func']:
                    func_decl, return_decl = [x.strip() for x in func['func'].split('->')]
                else:
                    raise Exception('Expected return declaration')
                    
                # 解析函数名和参数列表
                fn_name, arguments = func_decl.split('(')
                arguments = arguments.split(')')[0]
                
                # 构建函数声明字典
                declaration['name'] = func.get('name', fn_name)
                declaration['inplace'] = re.search('(^__i|[^_]_$)', fn_name) is not None
                return_arguments = parse_return_arguments(return_decl, declaration['inplace'])
                arguments = parse_arguments(arguments, func, declaration['name'], return_arguments)
                
                # 处理输出参数
                output_arguments = [x for x in arguments if x.get('output')]
                declaration['return'] = return_arguments if len(output_arguments) == 0 else output_arguments
                
                # 设置其他属性
                declaration['variants'] = func.get('variants', ['function'])
                declaration['requires_tensor'] = func.get('requires_tensor', False)
                declaration['cpu_half'] = func.get('cpu_half', False)
                declaration['deprecated'] = func.get('deprecated', False)
                declaration['device_guard'] = func.get('device_guard', True)
                declaration['arguments'] = func.get('arguments', arguments)
                declaration['type_method_definition_dispatch'] = func.get('dispatch', declaration['name'])
                declaration['python_module'] = func.get('python_module', '')
                declaration['aten_sparse'] = has_sparse_dispatches(
                    declaration['type_method_definition_dispatch'])
                    
                declarations.append(declaration)
            except Exception as e:
                # 错误处理：打印详细错误信息
                msg = '''Exception raised in processing function:
{func}
Generated partial declaration:
{decl}'''.format(func=pprint.pformat(func), decl=pprint.pformat(declaration))
                print(msg, file=sys.stderr)
                raise e

    return declarations