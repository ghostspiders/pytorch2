import re
from copy import deepcopy
from function_wrapper import TYPE_FORMAL_GENERIC
import common_with_cwrap

# 定义类型映射字典，将类型分类为浮点型和整型
type_map = {
    'floating_point': [  # 浮点类型
        'Float',
        'Double',
        'Half',
    ],
    'integral': [  # 整型
        'Byte',
        'Char',
        'Short',
        'Int',
        'Long'
    ],
}

# 合并所有类型
all_types = type_map['floating_point'] + type_map['integral']
type_map['all'] = all_types  # 添加'all'键包含所有类型

# 定义所有后端类型和默认后端
all_backends = ['CPU', 'CUDA', 'SparseCPU', 'SparseCUDA']
default_backends = ['CPU', 'CUDA']

# 定义稀疏后端映射
sparse_map = {
    'CPU': 'SparseCPU',
    'CUDA': 'SparseCUDA',
}


def process_types_and_backends(option):
    """处理类型和后端组合"""
    # 如果没有指定具体的后端类型组合，则根据后端和类型属性枚举
    if 'backend_type_pairs' not in option:
        # 获取后端列表，默认为default_backends
        backends = option.get('backends', default_backends)
        # 如果需要稀疏张量支持，添加对应的稀疏后端
        if option.get('aten_sparse', False):
            backends.extend([sparse_map[p] for p in backends if p in sparse_map])
        backends = set(backends)  # 去重

        # 获取类型列表，默认为所有类型
        types = option.get('types', all_types)

        # 生成所有后端和类型的组合
        pairs = [[p, t] for p in backends for t in types]
    else:
        pairs = option['backend_type_pairs']

    # 展开类型别名（如integral, floating_point, all）
    def expand(pair):
        p, t = pair
        assert(p in all_backends)
        if t in type_map:  # 如果是类型别名
            return [(p, tt) for tt in type_map[t]]  # 展开为具体类型
        assert(t in all_types)
        return [(p, t)]
    pairs = set(p for pair in pairs for p in expand(pair))

    # 如果有稀疏张量参数，禁用CUDA Half类型
    for arg in option.get('arguments', []):
        if arg['type'] == 'THSTensor*':
            pairs.discard(('CUDA', 'Half'))

    # 特殊处理：除非明确启用，否则移除CPU Half类型
    if not option.get('cpu_half', False):
        pairs.discard(('CPU', 'Half'))

    # 对结果排序以便阅读
    option['backend_type_pairs'] = sorted([p for p in pairs])


def exclude(declaration):
    """判断是否排除该声明"""
    return 'only_register' in declaration or declaration.get('name') == 'ndimension'


def add_variants(option):
    """添加变体，默认为method变体"""
    option.setdefault('variants', ['method'])


def handle_outputs_taken_as_arguments(options):
    """处理作为参数输出的情况"""
    new_options = []

    def is_nullable(arg):
        """判断参数是否可为空"""
        return (arg['type'] in {'THIntegerTensor*', 'THTensor*'} and
                arg.get('default', '') in {None, 'NULL', 'nullptr'})

    def should_generate_out_variant(option):
        """判断是否应该生成_out变体"""
        if 'function' in option['variants'] and option['mode'] != 'native':
            # 不生成原地操作的_out变体
            return re.search('(^__i|[^_]_$)', option['api_name']) is None
        return False

    for option in options:
        # 标记可为空的参数
        for arg in option['arguments']:
            if is_nullable(arg):
                arg['is_nullable'] = True

        # 处理有输出参数的情况
        if any('output' in arg for arg in option['arguments']):
            allocate_option = deepcopy(option)
            # 标记分配内存的选项
            for arg in allocate_option['arguments']:
                if 'output' in arg:
                    arg['allocate'] = True

            # 原始选项不再作为方法，并添加_out后缀表示接受输出参数
            if should_generate_out_variant(option):
                if 'method' in option['variants']:
                    option['variants'].remove('method')
                option['api_name'] += '_out'
                new_options.append(option)

            new_options.append(allocate_option)
        else:
            new_options.append(option)
    return new_options


def sanitize_return(option):
    """规范化返回类型"""
    ret = option['return']
    m = re.match(r'argument (\d+(,\d+)*)', ret)
    if m is not None:
        # 返回类型为参数索引
        arguments = [int(x) for x in m.group(1).split(',')]
        option['return'] = {'kind': 'arguments', 'arguments': arguments}
    elif ret == 'self':
        # 返回类型为self参数
        option['return'] = {'kind': 'arguments', 'arguments': []}
        for i, x in enumerate(option['arguments']):
            if x['name'] == 'self':
                option['return']['arguments'].append(i)
                break
    else:
        # 返回类型为具体类型
        option['return'] = {'kind': 'type', 'type': option['return']}


def set_mode(option):
    """设置模式，默认为TH模式"""
    option['mode'] = option.get('mode', 'TH')


def discover_zero_dim_tensor_operations(declaration):
    """发现0维张量操作"""
    def exclude(arg):
        return arg.get('ignore_check')

    def signature(option, i=None, value=None):
        """生成函数签名"""
        elements = [TYPE_FORMAL_GENERIC.get(arg['type'], arg['type'])
                    if i is None or j != i else value
                    for j, arg in enumerate(option['arguments'])
                    if not exclude(arg)]
        return '#'.join(elements)
    
    # 建立签名到选项的映射
    signature_to_option = {signature(option): option
                           for option in declaration['options']}

    # 查找可以用0维张量替代标量的操作
    for option in declaration['options']:
        for i, arg in enumerate(option['arguments']):
            if arg['type'] == 'real':  # 标量类型
                signature_of_tensor_version = signature(option, i, 'Tensor &')
                if signature_of_tensor_version in signature_to_option:
                    tensor_version = signature_to_option[signature_of_tensor_version]
                    names = [arg['name'] for arg in tensor_version['arguments']
                             if not exclude(arg)]
                    # 标记可以用0维张量替代标量的参数
                    tensor_version['zero_dim_dispatch_when_scalar'] = names[i]


def discover_sparse_tensor_operations(declaration):
    """发现稀疏张量操作"""
    def exclude(arg):
        return arg.get('ignore_check')

    def signature(option, i=None, value=None):
        """生成函数签名"""
        elements = [TYPE_FORMAL_GENERIC.get(arg['type'], arg['type'])
                    if i is None or j != i else value
                    for j, arg in enumerate(option['arguments'])
                    if not exclude(arg)]
        return '#'.join(elements)

    # 查找有'aten_dense_sparse'标志的选项
    dense_sparse_options = [option
                            for option in declaration['options']
                            if option.get('aten_dense_sparse', False)]
    if len(dense_sparse_options) > 0:
        signature_to_option = {signature(option): option
                               for option in declaration['options']}

        # 查找可以用稀疏张量替代密集张量的操作
        for option in declaration['options']:
            for i, arg in enumerate(option['arguments']):
                if (arg['type'] == 'THSTensor*' and
                        option.get('aten_dense_sparse', False)):
                    signature_of_tensor_version = signature(
                        option, i, 'Tensor &')
                    if signature_of_tensor_version in signature_to_option:
                        tensor_version = signature_to_option[signature_of_tensor_version]
                        raw_args = len(tensor_version['arguments'])
                        names = [arg['name'] for arg in tensor_version['arguments']
                                 if not exclude(arg)]
                        filtered_args = len(names)
                        # 标记可以用稀疏张量替代密集张量的参数
                        tensor_version['when_sparse_dispatch'] = names[i -
                                                                       (raw_args - filtered_args)]


def is_extended_method(option):
    """判断是否为扩展方法"""
    if 'method' in option['variants']:
        return False
    else:
        return True


def run(declarations):
    """主处理函数"""
    # 过滤掉需要排除的声明
    declarations = [d for d in declarations if not exclude(d)]
    non_extended_methods = set()
    
    for declaration in declarations:
        # 设置声明默认值
        common_with_cwrap.set_declaration_defaults(declaration)
        # 深拷贝选项
        declaration['options'] = [deepcopy(o) for o in declaration['options']]
        # 过滤唯一选项
        declaration['options'] = common_with_cwrap.filter_unique_options(
            declaration['options'],
            allow_kwarg=False,
            type_to_signature=TYPE_FORMAL_GENERIC,
            remove_self=True)

        # 按选项数量排序
        common_with_cwrap.sort_by_number_of_options(declaration)

        # 发现0维张量和稀疏张量操作
        discover_zero_dim_tensor_operations(declaration)
        discover_sparse_tensor_operations(declaration)

        # 处理每个选项
        for option in declaration['options']:
            set_mode(option)
            if option['mode'] != 'native':
                sanitize_return(option)
            process_types_and_backends(option)
            add_variants(option)
            if not is_extended_method(option):
                non_extended_methods.add(option['api_name'])
        # 处理输出参数
        declaration['options'] = handle_outputs_taken_as_arguments(
            declaration['options'])
    
    # 处理重载的虚方法，将所有重载移动到Type中
    for declaration in declarations:
        for option in declaration['options']:
            option['extended_method'] = option['api_name'] not in non_extended_methods
    return declarations