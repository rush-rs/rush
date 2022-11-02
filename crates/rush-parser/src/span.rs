pub struct Span {
     pub start: Location,
     pub end: Location,
}

pub struct Location {
     pub line: usize,
     pub column: usize,
     pub char_idx: usize,
     pub byte_idx: usize,
}
