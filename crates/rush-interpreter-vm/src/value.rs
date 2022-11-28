use std::fmt::Display;

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
    Unit,
}

impl Value {
    pub(crate) fn into_int(self) -> i64 {
        match self {
            Value::Int(value) => value,
            _ => unreachable!("no illegal calls exist"),
        }
    }

    pub(crate) fn into_bool(self) -> bool {
        match self {
            Value::Bool(value) => value,
            _ => unreachable!("no illegal calls exist"),
        }
    }

    pub(crate) fn into_float(self) -> f64 {
        match self {
            Value::Float(value) => value,
            _ => unreachable!("no illegal calls exist"),
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
            Value::Char(_value) => todo!(), // TODO: is this correct?
            _ => unreachable!("never called this way"),
        }
    }

    pub(crate) fn add(&self, rhs: Value) -> Result<Value, RuntimeError> {
        match self {
            Value::Int(value) => match value.checked_add(rhs.into_int()) {
                Some(res) => Ok(Value::Int(res)),
                None => Err(RuntimeError::new(
                    RuntimeErrorKind::Arithmetic,
                    format!("{self} + {rhs} is illegal"),
                )),
            },
            Value::Float(value) => Ok(Value::Float(value + rhs.into_float())),
            _ => unreachable!("other types do not support this operation"),
        }
    }

    pub(crate) fn sub(&self, rhs: Value) -> Result<Value, RuntimeError> {
        match self {
            Value::Int(value) => match value.checked_sub(rhs.into_int()) {
                Some(res) => Ok(Value::Int(res)),
                None => Err(RuntimeError::new(
                    RuntimeErrorKind::Arithmetic,
                    format!("{self} - {rhs} is illegal"),
                )),
            },
            Value::Float(value) => Ok(Value::Float(value - rhs.into_float())),
            _ => unreachable!("other types do not support this operation"),
        }
    }

    pub(crate) fn mul(&self, rhs: Value) -> Result<Value, RuntimeError> {
        match self {
            Value::Int(value) => match value.checked_mul(rhs.into_int()) {
                Some(res) => Ok(Value::Int(res)),
                None => Err(RuntimeError::new(
                    RuntimeErrorKind::Arithmetic,
                    format!("{self} * {rhs} is illegal"),
                )),
            },
            Value::Float(value) => Ok(Value::Float(value * rhs.into_float())),
            _ => unreachable!("other types do not support this operation"),
        }
    }

    pub(crate) fn pow(&self, rhs: Value) -> Result<Value, RuntimeError> {
        match self {
            Value::Int(_) if rhs.into_int() > u32::MAX as i64 => Err(RuntimeError::new(
                RuntimeErrorKind::Arithmetic,
                format!("exponent larger than {}", u32::MAX),
            )),
            Value::Int(value) => match value.checked_pow(rhs.into_int() as u32) {
                Some(res) => Ok(Value::Int(res)),
                None => Err(RuntimeError::new(
                    RuntimeErrorKind::Arithmetic,
                    format!("{self} ** {rhs} is illegal"),
                )),
            },
            _ => unreachable!("other types do not support this operation"),
        }
    }

    pub(crate) fn div(&self, rhs: Value) -> Result<Value, RuntimeError> {
        match self {
            Value::Int(value) => match value.checked_div(rhs.into_int()) {
                Some(res) => Ok(Value::Int(res)),
                None => Err(RuntimeError::new(
                    RuntimeErrorKind::Arithmetic,
                    format!("{self} / {rhs} is illegal"),
                )),
            },
            Value::Float(value) => Ok(Value::Float(value / rhs.into_float())),
            _ => unreachable!("other types do not support this operation"),
        }
    }

    pub(crate) fn rem(&self, rhs: Value) -> Result<Value, RuntimeError> {
        match self {
            Value::Int(value) => match value.checked_rem(rhs.into_int()) {
                Some(res) => Ok(Value::Int(res)),
                None => Err(RuntimeError::new(
                    RuntimeErrorKind::Arithmetic,
                    format!("{self} % {rhs} is illegal"),
                )),
            },
            _ => unreachable!("other types do not support this operation"),
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
            _ => unreachable!("other types cannot be compared"),
        };
        Value::Bool(res)
    }

    pub(crate) fn le(&self, rhs: Value) -> Value {
        let res = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => *lhs <= rhs,
            (Value::Float(lhs), Value::Float(rhs)) => *lhs <= rhs,
            _ => unreachable!("other types cannot be compared"),
        };
        Value::Bool(res)
    }

    pub(crate) fn gt(&self, rhs: Value) -> Value {
        let res = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => *lhs > rhs,
            (Value::Float(lhs), Value::Float(rhs)) => *lhs > rhs,
            _ => unreachable!("other types cannot be compared"),
        };
        Value::Bool(res)
    }

    pub(crate) fn ge(&self, rhs: Value) -> Value {
        let res = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => *lhs >= rhs,
            (Value::Float(lhs), Value::Float(rhs)) => *lhs >= rhs,
            _ => unreachable!("other types cannot be compared"),
        };
        Value::Bool(res)
    }

    pub(crate) fn shl(&self, rhs: Value) -> Result<Value, RuntimeError> {
        let Value::Int(lhs) = self else {
            unreachable!("other types cannot be shifted");
        };
        if rhs.into_int() > u32::MAX as i64 {
            return Err(RuntimeError::new(
                RuntimeErrorKind::Arithmetic,
                format!("rhs larger than {}", u32::MAX),
            ));
        }
        match lhs.checked_shl(rhs.into_int() as u32) {
            Some(res) => Ok(Value::Int(res)),
            None => Err(RuntimeError::new(
                RuntimeErrorKind::Arithmetic,
                format!("{self} << {rhs} is illegal"),
            )),
        }
    }

    pub(crate) fn shr(&self, rhs: Value) -> Result<Value, RuntimeError> {
        let Value::Int(lhs) = self else {
            unreachable!("other types cannot be shifted");
        };
        if rhs.into_int() > u32::MAX as i64 {
            return Err(RuntimeError::new(
                RuntimeErrorKind::Arithmetic,
                format!("rhs larger than {}", u32::MAX),
            ));
        }
        match lhs.checked_shr(rhs.into_int() as u32) {
            Some(res) => Ok(Value::Int(res)),
            None => Err(RuntimeError::new(
                RuntimeErrorKind::Arithmetic,
                format!("{self} >> {rhs} is illegal"),
            )),
        }
    }

    pub(crate) fn bit_or(&self, rhs: Value) -> Value {
        match self {
            Value::Int(val) => Value::Int(val | rhs.into_int()),
            Value::Bool(val) => Value::Bool(val | rhs.into_bool()),
            _ => unreachable!("other types are illegal"),
        }
    }

    pub(crate) fn bit_and(&self, rhs: Value) -> Value {
        match self {
            Value::Int(val) => Value::Int(val & rhs.into_int()),
            Value::Bool(val) => Value::Bool(val & rhs.into_bool()),
            _ => unreachable!("other types are illegal"),
        }
    }

    pub(crate) fn bit_xor(&self, rhs: Value) -> Value {
        match self {
            Value::Int(val) => Value::Int(val ^ rhs.into_int()),
            Value::Bool(val) => Value::Bool(val ^ rhs.into_bool()),
            _ => unreachable!("other types are illegal"),
        }
    }

    pub(crate) fn and(&self, rhs: Value) -> Value {
        match self {
            Value::Bool(val) => Value::Bool(*val && rhs.into_bool()),
            _ => unreachable!("other types are illegal"),
        }
    }

    pub(crate) fn or(&self, rhs: Value) -> Value {
        match self {
            Value::Bool(val) => Value::Bool(*val || rhs.into_bool()),
            _ => unreachable!("other types are illegal"),
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
            _ => unreachable!("other combinations are impossible"),
        };
        Value::Int(res)
    }

    fn cast_bool(self) -> Value {
        let res = match self {
            Value::Int(val) => val != 0,
            Value::Char(val) => val != 0,
            Value::Float(val) => val != 0.0,
            _ => unreachable!("other combinations are impossible"),
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
            _ => unreachable!("other combinations are impossible"),
        };
        Value::Char(res)
    }

    fn cast_float(self) -> Value {
        let res = match self {
            Value::Int(val) => val as f64,
            Value::Bool(val) => val as u8 as f64,
            Value::Char(val) => val as f64,
            _ => unreachable!("other combinations are impossible"),
        };
        Value::Float(res)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(val) => write!(f, "{val}"),
            Value::Bool(val) => write!(f, "{val}"),
            Self::Char(val) => write!(f, "{val}"),
            Value::Float(val) => write!(
                f,
                "{val}{zero}",
                zero = if val.fract() == 0.0 { ".0" } else { "" }
            ),
            Value::Unit => write!(f, "()"),
        }
    }
}
