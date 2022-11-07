use std::fmt::Debug;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: Location,
    pub end: Location,
}

impl Span {
    pub fn new(start: Location, end: Location) -> Self {
        Self { start, end }
    }
}

impl Debug for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match f.alternate() {
            true => write!(
                f,
                "{}:{}..{}:{}",
                self.start.line, self.start.column, self.end.line, self.end.column
            ),
            false => write!(f, "{}..{}", self.start.byte_idx, self.end.byte_idx),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Location {
    pub line: usize,
    pub column: usize,
    pub char_idx: usize,
    pub byte_idx: usize,
}

impl Default for Location {
    fn default() -> Self {
        Self {
            line: 1,
            column: 1,
            char_idx: 0,
            byte_idx: 0,
        }
    }
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

    pub fn until(self, end: Location) -> Span {
        Span { start: self, end }
    }
}
