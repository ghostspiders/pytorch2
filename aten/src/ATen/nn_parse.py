import copy
import re
import common_with_cwrap
import yaml
from collections import OrderedDict, defaultdict

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# matches `name`, `params` in `name(params)`
NAME_PARAM_REGEX = r'(\w+)\((.*)\)'


def argument_to_declaration(param, func=None):
    """将参数声明字符串转换为结构化字典
    Args:
        param (str): 参数声明字符串，如 "Tensor input" 或 "int dim=1"
        func (dict, optional): 包含函数额外信息的字典，如默认初始化值
    Returns:
        dict: 解析后的参数信息字典
    """
    arg = {}
    # 分割类型和参数名（如 "Tensor input" -> ["Tensor", "input"]）
    arg['type'], name = param.split(' ')
    
    # 处理可空类型（如 "Tensor?"）
    if (arg['type'].endswith('?')):
        arg['is_nullable'] = True
        arg['type'] = arg['type'].rstrip('?')  # 移除问号
    
    # 特殊类型转换规则
    if arg['type'] == 'Tensor':
        arg['type'] = 'THTensor*'       # PyTorch底层张量类型
    elif arg['type'] == 'LongTensor':
        arg['type'] = 'THIndexTensor*'   # 索引张量类型
    elif arg['type'] == 'Scalar':
        arg['type'] = 'accreal'         # 累加计算使用的实数类型
    elif arg['type'] == 'Generator*':
        arg['type'] = 'THGenerator*'     # 随机数生成器
    
    # 处理固定大小的IntList（如 "IntList[2]"）
    match = re.match(r'IntList\[(\d+)\]', arg['type'])
    if match:
        arg['type'] = 'IntList'
        arg['size'] = int(match.group(1))  # 提取尺寸
    
    # 处理带默认值的参数（如 "dim=1"）
    if '=' in name:
        name, default = name.split('=')
        arg['optional'] = True
        arg['default'] = default
    
    arg['name'] = name  # 最终参数名
    
    # 处理函数特定的附加属性
    if func is not None:
        default_inits = func.get('default_init', {})  # 延迟初始化的默认值
        wrap_dims = func.get('wrap_dim', {})         # 需要维度包装的参数
        
        if name in default_inits:
            arg['default_init'] = default_inits[name]  # 非constexpr默认值
        if name in wrap_dims:
            arg['wrap_dim'] = wrap_dims[name]         # 维度包装标记
    
    return arg


def output_arguments(thnn_function):
    """识别并格式化输出参数
    Args:
        thnn_function (THNNFunction): THNN函数对象
    Returns:
        list: 输出参数描述字典列表
    """
    cname = thnn_function.name
    output_args = []

    # 类型转换：将CUDA类型转为CPU类型（THCTensor* -> THTensor*）
    def map_to_th_type(t):
        if t.startswith('THC'):
            t = t.replace('THC', 'TH')  # 移除CUDA标识
        return t

    def is_output_arg(arg_name, func_name):
        """判断参数是否为输出参数"""
        # 更新输出类的参数
        if arg_name == 'output' and 'updateOutput' in cname:
            return True
        # 梯度类参数
        if name in {'gradInput', 'gradWeight', 'gradBias', 'gradGrid'}:
            return True
        # 池化操作的indices是输出（但Unpooling是输入）
        if arg_name == 'indices' and 'updateOutput' in cname and 'Unpool' not in cname:
            return True
        return False

    # 遍历所有参数
    for arg in thnn_function.arguments:
        name = arg.name
        if is_output_arg(name, cname):
            desc = {
                'type': map_to_th_type(arg.type),  # 类型转换
                'name': camel_to_snake(name),      # 转蛇形命名
                'output': True,                    # 标记为输出
            }
            # 梯度参数可为空
            if name.startswith('grad_'):
                desc['is_nullable'] = True
            output_args.append(desc)
    
    return output_args


def get_return(args):
    """获取输出参数的索引位置
    Args:
        args (list): 参数列表，每个元素是包含参数信息的字典
    Returns:
        str: 返回格式化的输出参数索引字符串，如 "argument 0,1"
    """
    # 找出所有标记为output的参数索引，并转为字符串列表
    indices = [str(idx) for idx, arg in enumerate(args) if arg.get('output')]
    # 格式化为"argument 0,1,2"的形式
    return 'argument {}'.format(','.join(indices))


# 参数名称映射表（短名称->标准名称）
ARGUMENT_MAPPINGS = {
    'k': 'kernel_size',       # 卷积核尺寸
    'd': 'stride',            # 步长
    'pad': 'padding',         # 填充
    'p': 'padding',           # 填充(简写)
    'o': 'output_size',       # 输出尺寸
    'osize': 'output_size',   # 输出尺寸(全称)
    'output': 'output_size',  # 作为前缀使用(如outputW)
    'isize': 'input_size',    # 输入尺寸
    'dilation': 'dilation',   # 空洞率
    'adj': 'output_padding',  # 输出填充
    'a': 'output_padding',    # 输出填充(简写)
}

# 维度偏移量映射表（用于处理不同维度的参数）
DIMENSION_OFFSET = {
    # 空间维度
    'width': -1,    # 宽度(最后一维)
    'height': -2,   # 高度(倒数第二维)
    
    # 张量维度(Batch/Channel/Width/Height/Time)
    'B': 0,         # Batch维度
    'C': 1,         # Channel维度
    'W': -1,        # Width维度(同width)
    'H': -2,        # Height维度(同height)
    'T': -3,        # Time维度(3D数据)
    
    # 多方向填充
    'left': 0,      # 左填充
    'right': 1,     # 右填充
    'top': 2,       # 上填充
    'bottom': 3,    # 下填充
    'front': 4,     # 前填充(3D)
    'back': 5,      # 后填充(3D)
}

# 参数名称替换表（旧名称->新名称）
SUBSTITUTIONS = {
    'input': 'self',               # 将input改为self(更符合Python习惯)
    'weights': 'weight',           # 统一使用单数形式
    'train': 'training',           # 更明确的训练标志名称
    'val': 'value',                # 避免缩写
    'lambda': 'lambd',             # 避免Python关键字冲突
    'negval': 'negative_slope',    # 更清晰的负斜率命名
}

def camel_to_snake(name):
    """将驼峰命名转换为蛇形命名（例如：Conv2d -> conv2d）
    来源：https://stackoverflow.com/questions/1175208/
    Args:
        name (str): 驼峰命名的字符串
    Returns:
        str: 蛇形命名的字符串（全小写，下划线分隔）
    """
    # 处理大写字母后跟小写字母的情况（如：MyClass -> my_class）
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # 处理小写字母/数字后跟大写字母的情况（如：myVar -> my_var）
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_thnn_args(thnn_function, params, inplace):
    """将THNN函数参数转换为实际调用参数
    Args:
        thnn_function (THNNFunction): THNN函数对象
        params (list): 参数字典列表
        inplace (bool): 是否就地操作
    Returns:
        list: 转换后的参数列表
    Raises:
        RuntimeError: 当找不到参数绑定时抛出异常
    """
    # 构建参数名到参数的映射字典
    params_by_name = {p['name']: p for p in params}

    def arg_expr(prefix, suffix):
        """处理维度参数表达式（如kW -> kernel_size[0]）
        Args:
            prefix (str): 参数前缀（如'k'对应'kernel_size'）
            suffix (str): 维度后缀（如'W'对应宽度维度）
        Returns:
            dict: 表达式类型的参数描述字典
        """
        # 通过映射表获取标准参数名
        name = ARGUMENT_MAPPINGS[prefix]
        if name not in params_by_name:
            raise RuntimeError('missing arg "{}" in {}'.format(name, thnn_function.name))
        
        param = params_by_name[name]
        # 处理IntList类型且指定尺寸的情况
        if param['type'] == 'IntList' and 'size' in param:
            name = name + '_'  # 添加后缀避免冲突
        
        # 计算实际维度索引（支持正负索引）
        index = DIMENSION_OFFSET[suffix]
        if index < 0:
            index += param['size']
        
        # 返回表达式参数（如"kernel_size[0]"）
        return {'type': 'EXPRESSION', 'name': '{}[{}]'.format(name, index)}

    thnn_args = []
    for arg in thnn_function.arguments:
        name = arg.name
        # 跳过特殊参数'state'
        if name == 'state':
            continue
        
        # 处理就地操作的output参数
        if inplace and name == 'output':
            name = 'self'
        
        # 统一参数命名（蛇形命名 + 替换表处理）
        aten_name = camel_to_snake(SUBSTITUTIONS.get(name, name))
        parts = aten_name.split('_')  # 分割下划线分隔的部分
        
        # 直接匹配参数名的情况
        if aten_name in params_by_name:
            param = params_by_name[aten_name]
            # 保留可选参数标记
            if arg.is_optional:
                param['is_nullable'] = True
            thnn_args.append(copy.deepcopy(param))
        
        # 处理带维度后缀的参数（如pad_left）
        elif len(parts) == 2 and parts[0] in ARGUMENT_MAPPINGS and parts[1] in DIMENSION_OFFSET:
            thnn_args.append(arg_expr(parts[0], parts[1]))
        
        # 处理单字母维度参数（如kW -> kernel_size[0]）
        elif name[-1] in DIMENSION_OFFSET and name[:-1] in ARGUMENT_MAPPINGS:
            thnn_args.append(arg_expr(name[:-1], name[-1]))
        
        # 处理特殊尺寸参数（如owidth -> output_size[0]）
        elif name == 'owidth' or name == 'oheight':
            thnn_args.append(arg_expr(name[0], name[1:]))
        
        # 处理scale参数（固定为1）
        elif name == 'scale':
            thnn_args.append({'type': 'EXPRESSION', 'name': '1'})
        
        # 处理inplace参数（使用传入的布尔值）
        elif name == 'inplace':
            thnn_args.append({'type': 'EXPRESSION', 'name': str(inplace).lower()})
        
        # 无法识别的参数抛出异常
        else:
            raise RuntimeError("{}: can't find binding for '{}'"
                               .format(thnn_function.name, name))
    
    return thnn_args


def remove_unused_args(args, thnn_args):
    """移除未被THNN函数使用的参数
    Args:
        args (list): 原始参数列表
        thnn_args (list): THNN函数实际使用的参数列表
    Returns:
        list: 过滤后的参数列表
    """
    def clean_name(name):
        """清理参数名（移除数组下标和尾部下划线）"""
        name = name[:name.index('[')] if '[' in name else name
        if name.endswith('_'):
            name = name[:-1]
        return name
    
    # 获取THNN实际使用的参数名集合（自动添加output_mask）
    uses = set([clean_name(arg['name']) for arg in thnn_args])
    uses.add('output_mask')
    
    # 过滤参数并移除默认值（避免干扰调用）
    args = [arg for arg in args if arg['name'] in uses]
    for arg in args:
        if 'default' in arg:
            del arg['default']
    return args


def unique_args(argslist):
    """合并多个参数列表并去重
    Args:
        argslist (list): 包含多个参数列表的列表
    Returns:
        list: 去重后的合并参数列表
    """
    result = []
    seen = set()
    for args in argslist:
        for arg in args:
            if arg['name'] in seen:
                continue
            seen.add(arg['name'])
            result.append(arg)
    return result


def function_info(name, arguments, cimpls, buffers, backends, inplace, scalar_check):
    """构建函数信息字典
    Args:
        name (str): 函数名
        arguments (list): 参数列表
        cimpls (list): C实现配置列表
        buffers (list): 缓冲区参数列表
        backends (list): 支持的backend类型
        inplace (bool): 是否就地操作
        scalar_check (dict): 标量检查配置
    Returns:
        dict: 完整的函数描述字典
    """
    return {
        'mode': 'NN',  # 标记为神经网络函数
        'name': name,
        'types': ['Float', 'Double', 'Half'],  # 支持的数据类型（Half在CPU backend会被移除）
        'arguments': arguments,
        'return': 'argument 0' if inplace else get_return(arguments),  # 返回规则
        'buffers': buffers,
        'backends': backends,  # 支持的backend（CPU/CUDA）
        'cimpls': cimpls,  # C实现配置
        'scalar_check': scalar_check,  # 标量检查配置
        'variants': ['function'],  # 函数变体类型
    }


def base_declaration(func, thnn_function, backends, inplace=False):
    """创建基础函数声明（不包含缓冲区参数）
    Args:
        func (dict): 函数配置字典
        thnn_function (THNNFunction): THNN函数对象
        backends (list): 支持的backend类型
        inplace (bool): 是否就地操作
    Returns:
        dict: 基础函数信息字典
    """
    # 解析函数名和参数
    name, params = re.match(NAME_PARAM_REGEX, func['name']).groups()
    if inplace:
        name += '_'  # 就地操作添加后缀
    
    # 转换参数声明
    params = params.split(', ')
    arguments = [argument_to_declaration(a, func) for a in params]
    
    # 非就地操作添加输出参数
    if not inplace:
        arguments += output_arguments(thnn_function)
    
    # 处理缓冲区参数
    buffers = [argument_to_declaration('Tensor ' + buf)
               for buf in func.get('buffers', [])]

    return function_info(name, arguments, None, buffers, backends, inplace, func.get('scalar_check'))


def forward_declaration(base, thnn_function, inplace=False):
    """创建前向传播函数声明
    Args:
        base (dict): 基础函数信息
        thnn_function (THNNFunction): THNN函数对象
        inplace (bool): 是否就地操作
    Returns:
        dict: 前向传播函数信息字典
    """
    # 构建函数名
    name = '{}_forward'.format(base['name'])
    if inplace:
        name += '_'

    # 复制非输出参数
    arguments = [copy.deepcopy(arg) for arg in base['arguments']
                 if not arg.get('output')]
    
    # 添加输出参数和缓冲区
    arguments += output_arguments(thnn_function)
    for buffer in base['buffers']:
        buffer = copy.deepcopy(buffer)
        buffer['output'] = True  # 标记缓冲区为输出
        arguments.append(buffer)

    # 获取THNN参数并过滤未使用的参数
    thnn_args = get_thnn_args(thnn_function, arguments, inplace)
    arguments = remove_unused_args(arguments, thnn_args)
    
    # 构建C实现配置
    cimpl = {'cname': thnn_function.name, 'arguments': thnn_args}

    # 过滤输出参数的标量检查配置
    scalar_check = base['scalar_check']
    if scalar_check is not None:
        output_arg_names = [arg['name'] for arg in arguments if arg.get('output', False)]
        scalar_check = {k: v for (k, v) in scalar_check.items() if k in output_arg_names}

    return function_info(name, arguments, [cimpl], [], base['backends'], inplace, scalar_check)


def backward_declaration(base, thnn_functions):
    """创建反向传播函数声明
    Args:
        base (dict): 基础函数信息
        thnn_functions (list): THNN函数对象列表
    Returns:
        dict: 反向传播函数信息字典
    """
    name = '{}_backward'.format(base['name'])

    # 构建基础参数列表（梯度输出+原始输入参数）
    arguments = []
    arguments.append({'type': 'THTensor*', 'name': 'grad_output'})  # 梯度输入
    arguments += [copy.deepcopy(arg) for arg in base['arguments']
                  if arg['name'] != 'inplace']  # 排除inplace参数
    arguments += base['buffers']  # 添加缓冲区

    # 特殊处理上采样操作的input_size参数
    if 'upsample' in base['name']:
        size = 2 + int(re.search(r'(\d+)d', base['name']).group(1))  # 计算输入维度
        input_size_arg = {'type': 'IntList', 'name': 'input_size', 'size': size}
        for output_size_idx, arg in enumerate(arguments):
            if arg['name'] == 'output_size':
                break
        arguments.insert(output_size_idx + 1, input_size_arg)  # 在output_size后插入

    # 清理前向传播的输出标记
    for arg in arguments:
        if 'output' in arg:
            del arg['output']

    # 添加THNN函数的输出参数（去重）
    arguments += unique_args([output_arguments(f) for f in thnn_functions])

    def initialize_output_arg(arg):
        """初始化输出参数配置"""
        arg['mask'] = True  # 标记参与output_mask计算
        arg['is_nullable'] = True  # 允许为空

        # 梯度参数的特殊处理
        if arg['name'] == 'grad_weight':
            arg['resize'] = 'weight'  # 根据weight调整大小
            arg['zero'] = True  # 需要零初始化
        if arg['name'] == 'grad_bias':
            dim = 1 if 'transpose' in name else 0  # 判断维度
            arg['resize'] = [('weight', dim)]  # 根据weight的特定维度调整
            arg['zero'] = True

    # 特殊处理批归一化的反向传播
    is_batch_norm_backward = '_backward' in thnn_functions[0].name
    grad_params = []
    if len(thnn_functions) > 1 or is_batch_norm_backward:
        for arg in arguments:
            if arg.get('output', False):
                initialize_output_arg(arg)  # 初始化输出参数
            # 收集梯度参数名（用于条件判断）
            if 'Tensor' in arg['type'] and arg['name'].startswith('grad_') and \
                    'input' not in arg['name'] and 'output' not in arg['name']:
                grad_params.append(arg['name'])

    # 获取THNN参数并过滤未使用的参数
    thnn_args = [get_thnn_args(f, arguments, False) for f in thnn_functions]
    arguments = remove_unused_args(arguments, unique_args(thnn_args))
    cimpls = []

    def get_condition(func):
        """生成THNN调用的条件判断"""
        if '_updateGradInput' in func.name:
            return 'grad_input_'  # 检查grad_input是否非空
        if '_accGradParameters' in func.name:
            return ' || '.join(p + '_' for p in grad_params)  # 检查任意梯度参数非空
        return None

    # 构建每个THNN函数的C实现配置
    for func, args in zip(thnn_functions, thnn_args):
        cimpl = {'cname': func.name, 'arguments': args}
        if len(thnn_functions) > 1:
            cimpl['condition'] = get_condition(func)  # 添加条件判断
        cimpls.append(cimpl)

    # 处理标量检查配置
    output_args = [arg for arg in arguments if arg.get('output', False)]
    scalar_check_arg = base['scalar_check'] if base['scalar_check'] is not None else dict()
    scalar_check = {k: v for (k, v) in scalar_check_arg.items() if k in [a['name'] for a in output_args]}
    
    # 自动推断未指定的标量检查
    for arg in output_args:
        if scalar_check.get(arg['name']) is not None or arg.get('resize', False):
            continue
        base_name = arg['name'][len('grad_'):] if arg['name'] != 'grad_input' else 'self'
        if base_name in [a['name'] for a in arguments]:
            scalar_check[arg['name']] = base_name + '_->dim() == 0'  # 自动生成维度检查
        else:
            raise ValueError(f"无法推断 {arg['name']} 的标量检查，因为 {base_name} 不存在")

    return function_info(name, arguments, cimpls, [], base['backends'], False, scalar_check)
def parse_nn_yaml(filename):
    """解析神经网络YAML配置文件
    Args:
        filename (str): YAML文件路径
    Returns:
        dict: 解析后的神经网络函数配置字典
    """
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=Loader)  # 使用安全的YAML加载器


# 正则表达式匹配规则：
include_only = '(updateOutput|updateGradInput|accGradParameters|backward)$'  # 包含的函数后缀
exclude = 'LookupTable'  # 排除的函数名


def run(paths):
    """主处理函数，生成所有神经网络函数声明
    Args:
        paths (list): 包含.h和.yaml文件路径的列表
    Returns:
        list: 生成的函数声明字典列表
    """
    # 初始化数据结构
    function_backends = defaultdict(list)  # 记录函数支持的backend类型
    header_functions = OrderedDict()  # 保存从头文件解析的函数信息

    # 分离头文件和YAML文件
    headers = [p for p in paths if p.endswith('.h')]
    yamls = [p for p in paths if p.endswith('.yaml')]

    # 解析头文件
    for path in headers:
        # 确定backend类型（根据文件名判断CUDA或CPU）
        backend = 'CUDA' if re.search('THCU', path) else 'CPU'
        
        # 解析头文件中的函数声明
        for func in common_with_cwrap.parse_header(path):
            # 过滤不符合条件的函数
            if re.search(include_only, func.name) is None or re.search(exclude, func.name) is not None:
                continue
            
            # 记录函数backend信息
            function_backends[func.name].append(backend)
            if func.name not in header_functions:
                header_functions[func.name] = func  # 保存函数对象

    # 反向传播函数后缀列表
    bwd_suffixes = ['_updateGradInput', '_accGradParameters', '_backward']

    # 生成函数声明
    declarations = []
    for path in yamls:
        # 解析YAML文件中的函数配置
        for func in parse_nn_yaml(path):
            cname = func['cname']  # 获取基础函数名
            
            # 获取支持的backend类型（基于updateOutput函数）
            backends = function_backends[cname + '_updateOutput']
            
            # 获取前向传播函数对象
            fwd_function = header_functions[cname + '_updateOutput']
            
            # 收集反向传播相关函数对象
            bwd_functions = []
            for suffix in bwd_suffixes:
                if cname + suffix in header_functions:
                    bwd_functions.append(header_functions[cname + suffix])

            # 生成基础声明和前向/反向声明
            base = base_declaration(func, fwd_function, backends)
            declarations.append(base)
            declarations.append(forward_declaration(base, fwd_function))
            declarations.append(backward_declaration(base, bwd_functions))

            # 处理就地操作版本（如果有标记）
            if func.get('has_inplace', False):
                declarations.append(base_declaration(func, fwd_function, backends, True))
                declarations.append(forward_declaration(base, fwd_function, True))

    return declarations