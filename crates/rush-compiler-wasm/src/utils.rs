use rush_analyzer::Type;

use crate::types;

pub trait Leb128 {
    /// Convert the given number to a `Vec<u8>` in unsigned LEB128 encoding
    fn to_uleb128(&self) -> Vec<u8> {
        let mut buf = vec![];
        self.write_uleb128(&mut buf);
        buf
    }

    /// Write the given number into a `Vec<u8>` in unsigned LEB128 encoding
    fn write_uleb128(&self, buf: &mut Vec<u8>);

    /// Convert the given number to a `Vec<u8>` in signed LEB128 encoding
    fn to_sleb128(&self) -> Vec<u8> {
        let mut buf = vec![];
        self.write_sleb128(&mut buf);
        buf
    }

    /// Write the given number into a `Vec<u8>` in signed LEB128 encoding
    fn write_sleb128(&self, buf: &mut Vec<u8>);
}

impl Leb128 for u64 {
    fn write_uleb128(&self, buf: &mut Vec<u8>) {
        if *self < 128 {
            buf.push(*self as u8);
            return;
        }
        leb128::write::unsigned(buf, *self).expect("writing to a Vec should never fail");
    }

    fn write_sleb128(&self, buf: &mut Vec<u8>) {
        if *self < 64 {
            buf.push(*self as u8);
            return;
        }
        leb128::write::signed(buf, *self as i64).expect("writing to a Vec should never fail");
    }
}

macro_rules! leb128_impl {
    ($($type:ty),* $(,)?) => {$(
        impl Leb128 for $type {
            fn write_uleb128(&self, buf: &mut Vec<u8>) {
                (*self as u64).write_uleb128(buf);
            }

            fn write_sleb128(&self, buf: &mut Vec<u8>) {
                (*self as u64).write_sleb128(buf);
            }
        }
    )*};
}

leb128_impl!(bool, i8, u8, i16, u16, i32, u32, i64, isize, usize);

/// Convert a [`Type`] into its WASM equivalent
pub fn type_to_byte(type_: Type) -> Option<u8> {
    match type_ {
        Type::Int(0) => Some(types::I64),
        Type::Float(0) => Some(types::F64),
        Type::Bool(0) => Some(types::I32),
        Type::Char(0) => Some(types::I32),
        Type::Unit | Type::Never => None,
        Type::Unknown => unreachable!("the analyzer guarantees one of the above to match"),
        _ => todo!(), // TODO: what to do with pointers?
    }
}
