use std::{
    cmp::Ordering,
    ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub},
};

use crate::value::Value;

impl Not for Value {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Value::Int(num) => (!num).into(),
            Value::Bool(bool) => (!bool).into(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl Neg for Value {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Value::Int(num) => (-num).into(),
            Value::Float(num) => (-num).into(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => left.wrapping_add(right).into(),
            (Value::Float(left), Value::Float(right)) => (left + right).into(),
            (Value::Char(left), Value::Char(right)) => (left.wrapping_add(right) & 0x7f).into(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => left.wrapping_sub(right).into(),
            (Value::Float(left), Value::Float(right)) => (left - right).into(),
            (Value::Char(left), Value::Char(right)) => (left.wrapping_sub(right) & 0x7f).into(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => left.wrapping_mul(right).into(),
            (Value::Float(left), Value::Float(right)) => (left * right).into(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl Div for Value {
    type Output = Result<Self, &'static str>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (_, Value::Int(0)) => Err("division by zero"),
            (Value::Int(left), Value::Int(right)) => Ok(left.wrapping_div(right).into()),
            (Value::Float(left), Value::Float(right)) => Ok((left / right).into()),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl Rem for Value {
    type Output = Result<Self, &'static str>;

    fn rem(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (_, Value::Int(0)) => Err("division by zero"),
            (Value::Int(left), Value::Int(right)) => Ok(left.wrapping_rem(right).into()),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl Value {
    pub fn pow(self, exp: Self) -> Self {
        match (self, exp) {
            (Value::Int(_), Value::Int(exp)) if exp < 0 => 0_i64.into(),
            (Value::Int(base), Value::Int(exp)) => {
                base.pow((exp).try_into().unwrap_or(u32::MAX)).into()
            }
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Value::Int(left), Value::Int(right)) => left.partial_cmp(right),
            (Value::Float(left), Value::Float(right)) => left.partial_cmp(right),
            (Value::Char(left), Value::Char(right)) => left.partial_cmp(right),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl Shl for Value {
    type Output = Result<Self, String>;

    fn shl(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(_), Value::Int(right)) if !(0..=63).contains(&right) => Err(format!(
                "tried to shift left by {right}, which is outside `0..=63`"
            )),
            (Value::Int(left), Value::Int(right)) => Ok(left.shl(right).into()),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl Shr for Value {
    type Output = Result<Self, String>;

    fn shr(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(_), Value::Int(right)) if !(0..=63).contains(&right) => Err(format!(
                "tried to shift right by {right}, which is outside `0..=63`"
            )),
            (Value::Int(left), Value::Int(right)) => Ok(left.shr(right).into()),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl BitOr for Value {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => (left | right).into(),
            (Value::Bool(left), Value::Bool(right)) => (left | right).into(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl BitAnd for Value {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => (left & right).into(),
            (Value::Bool(left), Value::Bool(right)) => (left & right).into(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}

impl BitXor for Value {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(left), Value::Int(right)) => (left ^ right).into(),
            (Value::Bool(left), Value::Bool(right)) => (left ^ right).into(),
            _ => unreachable!("the analyzer guarantees one of the above to match"),
        }
    }
}
