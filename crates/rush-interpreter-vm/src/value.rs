use std::fmt::{self, Display, Formatter};

use crate::{
    instruction::Type,
    vm::{RuntimeError, RuntimeErrorKind},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Char(u8),
    Float(f64),
    Ptr(Pointer),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pointer {
    Rel(isize),
    Abs(usize),
}

impl Display for Pointer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Pointer::Rel(offset) => write!(f, "*rel[{offset}]"),
            Pointer::Abs(addr) => write!(f, "*abs[{addr}]"),
        }
    }
}

impl Value {
    pub(crate) fn unwrap_int(self) -> i64 {
        match self {
            Value::Int(value) => value,
            _ => panic!("called `Value::unwrap_int` on a non-int value"),
        }
    }

    pub(crate) fn unwrap_bool(self) -> bool {
        match self {
            Value::Bool(value) => value,
            _ => panic!("called `Value::unwrap_bool` on a non-bool value"),
        }
    }

    pub(crate) fn unwrap_ptr(self) -> Pointer {
        match self {
            Value::Ptr(ptr) => ptr,
            _ => panic!("called `Value::unwrap_ptr` on a non-ptr value"),
        }
    }

    pub(crate) fn neg(&self) -> Value {
        match self {
            Value::Int(val) => Value::Int(-val),
            Value::Float(val) => Value::Float(-val),
            _ => unreachable!("never called this way"),
        }
    }

    pub(crate) fn not(&self) -> Value {
        match self {
            Value::Int(val) => Value::Int(!val),
            Value::Bool(value) => Value::Bool(!value),
            _ => unreachable!("never called this way"),
        }
    }

    pub(crate) fn add(&self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs.wrapping_add(rhs)),
            (Value::Char(lhs), Value::Char(rhs)) => Value::Char(lhs.wrapping_add(rhs) & 0x7f),
            (Value::Float(lhs), Value::Float(rhs)) => Value::Float(lhs + rhs),
            _ => unreachable!("other types do not support this operation: {self} + {rhs}"),
        }
    }

    pub(crate) fn sub(&self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs.wrapping_sub(rhs)),
            (Value::Char(lhs), Value::Char(rhs)) => Value::Char(lhs.wrapping_sub(rhs) & 0x7f),
            (Value::Float(lhs), Value::Float(rhs)) => Value::Float(lhs - rhs),
            _ => unreachable!("other types do not support this operation: {self} - {rhs}"),
        }
    }

    pub(crate) fn mul(&self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs.wrapping_mul(rhs)),
            (Value::Float(lhs), Value::Float(rhs)) => Value::Float(lhs * rhs),
            _ => unreachable!("other types do not support this operation: {self} * {rhs}"),
        }
    }

    pub(crate) fn pow(&self, rhs: Value) -> Value {
        Value::Int(self.unwrap_int().wrapping_pow(rhs.unwrap_int() as u32))
    }

    pub(crate) fn div(&self, rhs: Value) -> Result<Value, RuntimeError> {
        match (self, rhs) {
            (_, Value::Int(0)) => Err(RuntimeError::new(
                RuntimeErrorKind::Arithmetic,
                format!("{self} / {rhs} is illegal"),
            )),
            (Value::Int(lhs), Value::Int(rhs)) => Ok(Value::Int(lhs.wrapping_div(rhs))),
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Float(lhs / rhs)),
            _ => unreachable!("other types do not support this operation: {self} / {rhs}"),
        }
    }

    pub(crate) fn rem(&self, rhs: Value) -> Result<Value, RuntimeError> {
        match (self, rhs) {
            (_, Value::Int(0)) => Err(RuntimeError::new(
                RuntimeErrorKind::Arithmetic,
                format!("{self} % {rhs} is illegal"),
            )),
            (Value::Int(lhs), Value::Int(rhs)) => Ok(Value::Int(lhs.wrapping_rem(rhs))),
            _ => unreachable!("other types do not support this operation: {self} % {rhs}"),
        }
    }

    pub(crate) fn eq(&self, rhs: Value) -> Value {
        Value::Bool(*self == rhs)
    }

    pub(crate) fn ne(&self, rhs: Value) -> Value {
        Value::Bool(*self != rhs)
    }

    pub(crate) fn lt(&self, rhs: Value) -> Value {
        let res = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => *lhs < rhs,
            (Value::Float(lhs), Value::Float(rhs)) => *lhs < rhs,
            (Value::Char(lhs), Value::Char(rhs)) => *lhs < rhs,
            _ => unreachable!("other types cannot be compared: {self} < {rhs}"),
        };
        Value::Bool(res)
    }

    pub(crate) fn le(&self, rhs: Value) -> Value {
        let res = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => *lhs <= rhs,
            (Value::Float(lhs), Value::Float(rhs)) => *lhs <= rhs,
            (Value::Char(lhs), Value::Char(rhs)) => *lhs <= rhs,
            _ => unreachable!("other types cannot be compared: {self} <= {rhs}"),
        };
        Value::Bool(res)
    }

    pub(crate) fn gt(&self, rhs: Value) -> Value {
        let res = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => *lhs > rhs,
            (Value::Float(lhs), Value::Float(rhs)) => *lhs > rhs,
            (Value::Char(lhs), Value::Char(rhs)) => *lhs > rhs,
            _ => unreachable!("other types cannot be compared: {self} > {rhs}"),
        };
        Value::Bool(res)
    }

    pub(crate) fn ge(&self, rhs: Value) -> Value {
        let res = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => *lhs >= rhs,
            (Value::Float(lhs), Value::Float(rhs)) => *lhs >= rhs,
            (Value::Char(lhs), Value::Char(rhs)) => *lhs >= rhs,
            _ => unreachable!("other types cannot be compared: {self} >= {rhs}"),
        };
        Value::Bool(res)
    }

    pub(crate) fn shl(&self, rhs: Value) -> Result<Value, RuntimeError> {
        let (Value::Int(lhs), Value::Int(rhs)) = (self, rhs) else {
            unreachable!("other types cannot be shifted: {self} << {rhs}");
        };
        if !(0..=63).contains(&rhs) {
            return Err(RuntimeError::new(
                RuntimeErrorKind::Arithmetic,
                "rhs is not in range `0..=63`".to_string(),
            ));
        }
        Ok(Value::Int(lhs << rhs as u32))
    }

    pub(crate) fn shr(&self, rhs: Value) -> Result<Value, RuntimeError> {
        let (Value::Int(lhs), Value::Int(rhs)) = (self, rhs) else {
            unreachable!("other types cannot be shifted: {self} >> {rhs}");
        };
        if !(0..=63).contains(&rhs) {
            return Err(RuntimeError::new(
                RuntimeErrorKind::Arithmetic,
                "rhs is not in range `0..=63`".to_string(),
            ));
        }
        Ok(Value::Int(lhs >> rhs as u32))
    }

    pub(crate) fn bit_or(&self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs | rhs),
            (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(lhs | rhs),
            _ => unreachable!("other types are illegal: {self} | {rhs}"),
        }
    }

    pub(crate) fn bit_and(&self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs & rhs),
            (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(lhs & rhs),
            _ => unreachable!("other types are illegal: {self} & {rhs}"),
        }
    }

    pub(crate) fn bit_xor(&self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs ^ rhs),
            (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(lhs ^ rhs),
            _ => unreachable!("other types are illegal: {self} ^ {rhs}"),
        }
    }

    pub(crate) fn cast(self, to: Type) -> Value {
        match to {
            Type::Int => self.cast_int(),
            Type::Bool => self.cast_bool(),
            Type::Char => self.cast_char(),
            Type::Float => self.cast_float(),
        }
    }

    fn cast_int(self) -> Value {
        let res = match self {
            Value::Bool(val) => val as i64,
            Value::Char(val) => val as i64,
            Value::Float(val) => val as i64,
            _ => unreachable!("other combinations are impossible: {self} as int"),
        };
        Value::Int(res)
    }

    fn cast_bool(self) -> Value {
        let res = match self {
            Value::Int(val) => val != 0,
            Value::Char(val) => val != 0,
            Value::Float(val) => val != 0.0,
            _ => unreachable!("other combinations are impossible: {self} as bool"),
        };
        Value::Bool(res)
    }

    fn cast_char(self) -> Value {
        let res = match self {
            Value::Int(i64::MIN..=0) => 0,
            Value::Int(127..=i64::MAX) => 127,
            Value::Int(val) => val as u8,
            Value::Bool(val) => val as u8,
            Value::Float(val) if val < 0.0 => 0,
            Value::Float(val) if val > 127.0 => 127,
            Value::Float(val) => val as u8,
            _ => unreachable!("other combinations are impossible: {self} as char"),
        };
        Value::Char(res)
    }

    fn cast_float(self) -> Value {
        let res = match self {
            Value::Int(val) => val as f64,
            Value::Bool(val) => val as u8 as f64,
            Value::Char(val) => val as f64,
            _ => unreachable!("other combinations are impossible: {self} as float"),
        };
        Value::Float(res)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(val) => write!(f, "{val}"),
            Value::Bool(val) => write!(f, "{val}"),
            Value::Char(val) => write!(f, "{val}"),
            Value::Float(val) => write!(
                f,
                "{val}{zero}",
                zero = if val.fract() == 0.0 { ".0" } else { "" }
            ),
            Value::Ptr(ptr) => match ptr {
                Pointer::Rel(offset) => write!(f, "*rel[{offset}]"),
                Pointer::Abs(addr) => write!(f, "*abs[{addr}]"),
            },
        }
    }
}
