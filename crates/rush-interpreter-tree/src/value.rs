use std::{borrow::Cow, cell::RefCell, rc::Rc};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Char(u8),
    Bool(bool),
    Unit,
    Ptr(Rc<RefCell<Value>>),
}

impl Value {
    pub fn wrapped(self) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(self))
    }
}

#[derive(Clone, Debug)]
pub enum InterruptKind {
    Return(Value),
    Break,
    Continue,
    Error(Cow<'static, str>),
    Exit(i64),
}

impl InterruptKind {
    pub fn into_value(self) -> Result<Value, InterruptKind> {
        match self {
            Self::Return(val) => Ok(val),
            kind @ (Self::Error(_) | Self::Exit(_)) => Err(kind),
            _ => Ok(Value::Unit),
        }
    }
}

impl<S> From<S> for InterruptKind
where
    S: Into<Cow<'static, str>>,
{
    fn from(msg: S) -> Self {
        Self::Error(msg.into())
    }
}

//////////////////////////////////

macro_rules! from_impl {
    ($variant:ident, $type:ty) => {
        impl From<$type> for Value {
            fn from(val: $type) -> Self {
                Self::$variant(val)
            }
        }

        impl From<&$type> for Value {
            fn from(val: &$type) -> Self {
                Self::from(*val)
            }
        }

        impl From<&mut $type> for Value {
            fn from(val: &mut $type) -> Self {
                Self::from(*val)
            }
        }
    };
}

from_impl!(Int, i64);
from_impl!(Float, f64);
from_impl!(Char, u8);
from_impl!(Bool, bool);

//////////////////////////////////

macro_rules! unwrap_impl {
    ($variant:ident, $res:ty, $name:ident) => {
        pub fn $name(self) -> $res {
            match self {
                Self::$variant(val) => val,
                other => panic!(concat!("called ", stringify!($name), " on `{:?}`"), other),
            }
        }
    };
}

impl Value {
    unwrap_impl!(Int, i64, unwrap_int);
    unwrap_impl!(Bool, bool, unwrap_bool);
    unwrap_impl!(Ptr, Rc<RefCell<Value>>, unwrap_ptr);
}
