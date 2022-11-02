#[derive(Default, Debug, Clone, Copy)]
pub struct Span {
    pub start: Location,
    pub end: Location,
}


#[derive(Default, Debug, Clone, Copy)]
pub struct Location {
    pub line: usize,
    pub column: usize,
    pub char_idx: usize,
    pub byte_idx: usize,
}

impl Location {
    pub fn advance(&mut self, newline: bool, byte_idx_offset: usize) {
        self.char_idx += 1;
        self.byte_idx += byte_idx_offset;
        if newline {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
    }
}
