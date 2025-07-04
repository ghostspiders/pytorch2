#include <ATen/core/jit_type.h>  // 包含JIT类型系统的头文件
#include <iostream>              // 标准输入输出流

namespace c10 {

/**
 * 重载<<运算符用于输出Type对象
 * @param out 输出流
 * @param t 要输出的Type对象
 * @return 输出流引用
 */
std::ostream& operator<<(std::ostream & out, const Type & t) {
  // 处理CompleteTensorType类型
  if(auto value = t.cast<CompleteTensorType>()) {
    out << toString(value->scalarType()) << "(";  // 输出标量类型
    auto& sizes = value->sizes();    // 获取维度大小
    auto& strides = value->strides(); // 获取步长
    AT_ASSERT(sizes.size() == strides.size());
    
    // 输出每个维度信息
    for (size_t i = 0; i < sizes.size(); i++) {
      if (i > 0) out << ", ";
      out << sizes[i];  // 输出维度大小
      
      // 检查是否连续内存布局
      int64_t expected = i + 1 < sizes.size() ? sizes[i+1]*strides[i+1] : 1;
      if (strides[i] != expected) {
        out << "!";  // 标记非连续内存
      }
    }
    out << ")";
  } 
  // 处理TensorType类型（不完整类型）
  else if (auto value = t.cast<TensorType>()) {
    out << toString(value->scalarType()) << "(";
    for (int i = 0; i < value->dim(); ++i) {
      if (i > 0) out << ", ";
      out << "*";  // 用*表示未知维度
    }
    out << ")";
  }
  // 处理列表类型
  else if(t.kind() == TypeKind::ListType) {
    auto prim = t.cast<ListType>()->getElementType();
    out << *prim << "[]";  // 输出元素类型后跟[]
  }
  // 处理可选类型
  else if (t.kind() == TypeKind::OptionalType) {
    auto prim = t.cast<OptionalType>()->getElementType();
    out << *prim << "?";  // 输出元素类型后跟?
  }
  // 处理Future类型
  else if(t.kind() == TypeKind::FutureType) {
    auto elem = t.cast<FutureType>()->getElementType();
    out << "Future[" << *elem << "]";  // 输出Future[元素类型]
  }
  // 处理元组类型
  else if(auto tup = t.cast<TupleType>()) {
    out << "(";
    for(size_t i = 0; i < tup->elements().size(); ++i) {
      if(i > 0) out << ", ";
      out << *(tup->elements()[i]);  // 输出元组每个元素的类型
    }
    out << ")";
  }
  // 默认处理：调用str()方法
  else {
    out << t.str();
  }
  return out;
}

// 以下是各种类型的单例获取方法实现
DynamicTypePtr DynamicType::get() {
  static auto value = DynamicType::create();
  return value;
}

UndefinedTensorTypePtr UndefinedTensorType::get() {
  static auto value = UndefinedTensorType::create();
  return value;
}

NumberTypePtr NumberType::get() {
  static auto value = NumberType::create();
  return value;
}

IntTypePtr IntType::get() {
  static auto value = IntType::create();
  return value;
}

FloatTypePtr FloatType::get() {
  static auto value = FloatType::create();
  return value;
}

BoolTypePtr BoolType::get() {
  static auto value = BoolType::create();
  return value;
}

NoneTypePtr NoneType::get() {
  static auto value = NoneType::create();
  return value;
}

GeneratorTypePtr GeneratorType::get() {
  static auto value = GeneratorType::create();
  return value;
}

StringTypePtr StringType::get() {
  static auto value = StringType::create();
  return value;
}

DeviceObjTypePtr DeviceObjType::get() {
  static auto value = DeviceObjType::create();
  return value;
}

// 以下是容器类型的工厂方法
OptionalTypePtr OptionalType::ofTensor() {
  static auto value = OptionalType::create(DynamicType::get());
  return value;
}

ListTypePtr ListType::ofTensors() {
  static auto value = ListType::create(DynamicType::get());
  return value;
}

ListTypePtr ListType::ofInts() {
  static auto value = ListType::create(IntType::get());
  return value;
}

ListTypePtr ListType::ofFloats() {
  static auto value = ListType::create(FloatType::get());
  return value;
}

ListTypePtr ListType::ofBools() {
  static auto value = ListType::create(BoolType::get());
  return value;
}

/**
 * 从IValue推断类型
 * @param value 输入值
 * @return 推断出的类型指针
 */
TypePtr inferTypeFrom(const IValue& value) {
  if (value.isTensor()) {
    return CompleteTensorType::create(value.toTensor());
  } else if (value.isDouble()) {
    return FloatType::get();
  } else if (value.isInt()) {
    return IntType::get();
  } else if (value.isBool()) {
    return BoolType::get();
  } else if (value.isString()) {
    return StringType::get();
  } else if (value.isIntList()) {
    return ListType::ofInts();
  } else if (value.isTensorList()) {
    return ListType::ofTensors();
  } else if (value.isBoolList()) {
    return ListType::ofBools();
  } else if (value.isDoubleList()) {
    return ListType::ofFloats();
  } else if (value.isTuple()) {
    return TupleType::create(fmap(value.toTuple()->elements(), inferTypeFrom));
  } else if (value.isDevice()) {
    return DeviceObjType::get();
  }
  AT_ASSERTM(false, "Unhandled IValue kind in inferTypeFrom");
}

/**
 * 统一两种类型
 * @param t1 第一种类型
 * @param t2 第二种类型
 * @return 统一后的类型，如果无法统一则返回nullopt
 */
c10::optional<TypePtr> unifyTypes(const TypePtr& t1, const TypePtr& t2) {
  // 处理子类型关系
  if (t1->isSubtypeOf(t2)) return t2;
  if (t2->isSubtypeOf(t1)) return t1;

  // 处理DynamicType情况
  if (t1->isSubtypeOf(DynamicType::get()) && t2->isSubtypeOf(DynamicType::get())) {
    return static_cast<TypePtr>(DynamicType::get());
  }

  // 处理Optional类型
  if (t1->isSubtypeOf(NoneType::get()) && !t2->isSubtypeOf(NoneType::get())) {
    return OptionalType::create(t2);
  } else if (t2->isSubtypeOf(NoneType::get()) && !t1->isSubtypeOf(NoneType::get())) {
    return OptionalType::create(t1);
  }

  // 处理容器类型(List/Tuple)
  if (t1->cast<ListType>() && t2->cast<ListType>()) {
    auto unified_type = unifyTypes(t1->cast<ListType>()->getElementType(), 
                                 t2->cast<ListType>()->getElementType());
    return unified_type ? ListType::create(*unified_type) : c10::nullopt;
  } 
  else if(t1->cast<TupleType>() && t2->cast<TupleType>()) {
    auto tuple1 = t1->cast<TupleType>();
    auto tuple2 = t2->cast<TupleType>();
    if (tuple1->elements().size() != tuple2->elements().size()) {
      return c10::nullopt;
    }
    std::vector<TypePtr> elements;
    for (size_t i = 0; i < tuple1->elements().size(); i++) {
      if (auto elem = unifyTypes(tuple1->elements().at(i), tuple2->elements().at(i))) {
        elements.push_back(*elem);
      } else {
        return c10::nullopt;
      }
    }
    return static_cast<TypePtr>(TupleType::create(elements));
  }

  return c10::nullopt;
}

/**
 * 匹配类型变量
 * @param formal 形式类型
 * @param actual 实际类型
 * @param type_env 类型环境
 * @return 匹配结果
 */
MatchTypeReturn matchTypeVariables(TypePtr formal, TypePtr actual, TypeEnv& type_env) {
  MatchTypeReturn ret;
  if(!formal->hasFreeVariables()) {
    ret.type = formal;
    return ret;
  }

  // 处理变量类型
  if(auto vt = formal->cast<VarType>()) {
    auto it = type_env.find(vt->name());
    if(it == type_env.end()) {
      type_env[vt->name()] = actual;
      ret.type = actual;
      return ret;
    } else if(auto unified = unifyTypes(it->second, actual)) {
      type_env[vt->name()] = *unified;
      ret.type = *unified;
      return ret;
    }
    std::stringstream ss;
    ss << "type variable '" << vt->name() <<"' previously matched to type " 
       << it->second->str() << " is matched to type " << actual->str();
    ret.errMsg = ss.str();
    return ret;
  }
  // 处理列表类型
  else if(auto lt_formal = formal->cast<ListType>()) {
    if(auto lt_actual = actual->cast<ListType>()) {
      const auto innerType = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerType.type) return innerType;
      ret.type = ListType::create(*innerType.type);
      return ret;
    } else {
      ret.errMsg = "cannot match a list to " + actual->str();
      return ret;
    }
  }
  // 处理元组类型
  else if(auto tp_formal = formal->cast<TupleType>()) {
    if(auto tp_actual = actual->cast<TupleType>()) {
      if(tp_formal->elements().size() != tp_actual->elements().size()) {
        ret.errMsg = "cannot match tuples of mismatched size";
        return ret;
      }
      std::vector<TypePtr> elements;
      for(size_t i = 0; i < tp_formal->elements().size(); ++i) {
        const auto result = matchTypeVariables(
            tp_formal->elements()[i], tp_actual->elements()[i], type_env);
        if (!result.type) return result;
        elements.push_back(*result.type);
      }
      ret.type = TupleType::create(std::move(elements));
      return ret;
    } else {
      ret.errMsg = "cannot match a tuple to " + actual->str();
      return ret;
    }
  }
  // 处理Future类型
  else if (auto lt_formal = formal->cast<FutureType>()) {
    if (auto lt_actual = actual->cast<FutureType>()) {
      const auto innerType = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerType.type) return innerType;
      ret.type = FutureType::create(*innerType.type);
      return ret;
    } else {
      ret.errMsg = "cannot match a future to " + actual->str();
      return ret;
    }
  }
  // 处理Optional类型
  else if (auto opt_formal = formal->cast<OptionalType>()) {
    if (auto opt_actual = actual->cast<OptionalType>()) {
      const auto optionedType = matchTypeVariables(
          opt_formal->getElementType(), opt_actual->getElementType(), type_env);
      if (!optionedType.type) return optionedType;
      ret.type = OptionalType::create(*optionedType.type);
      return ret;
    } else if (!actual->isSubtypeOf(NoneType::get())) {
      return matchTypeVariables(opt_formal->getElementType(), actual, type_env);
    } else {
      ret.errMsg = "cannot match an Optional[T] to None, because there is no way to determine T from None.";
      return ret;
    }
  }

  AT_ERROR("unhandled free variable container: ", formal->str());
}

/**
 * 评估类型变量
 * @param type 包含变量的类型
 * @param type_env 类型环境
 * @return 具体化的类型
 */
CAFFE2_API TypePtr evalTypeVariables(TypePtr type, std::unordered_map<std::string, TypePtr>& type_env) {
  if(!type->hasFreeVariables()) return type;

  if(auto vt = type->cast<VarType>()) {
    auto it = type_env.find(vt->name());
    AT_ASSERTM(it != type_env.end(), "schema has unbound type variable '", vt->name(), "' in its return type");
    return it->second;
  } else {
    auto new_contained = fmap(type->containedTypes(), [&](TypePtr t) {
      return evalTypeVariables(t, type_env);
    });
    return type->withContained(std::move(new_contained));
  }
}

/**
 * 类型种类转字符串
 * @param kind 类型种类
 * @return 对应的字符串表示
 */
const char * typeKindToString(TypeKind kind) {
#define CASE_TYPE(T) case TypeKind::T: return #T;
  switch(kind) {
    C10_FORALL_TYPES(CASE_TYPE)
  }
#undef CASE_TYPE
  return "";
}

/**
 * 子类型检查
 * @param rhs 要检查的父类型
 * @return 当前类型是否是rhs的子类型
 */
bool Type::isSubtypeOf(const TypePtr rhs) const {
  if(auto rhs_ = rhs->cast<OptionalType>()) {
    return this->isSubtypeOf(rhs_->getElementType());
  }
  return *this == *rhs;
}

} // namespace c10