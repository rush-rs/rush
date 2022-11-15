use std::fmt::Debug;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Span<'src> {
    pub start: Location<'src>,
    pub end: Location<'src>,
}

impl<'src> Span<'src> {
    pub fn new(start: Location<'src>, end: Location<'src>) -> Self {
        Self { start, end }
    }

    pub fn dummy() -> Self {
        Self {
            start: Location::new(""),
            end: Location::new(""),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.start.byte_idx == self.end.byte_idx
    }
}

impl Debug for Span<'_> {
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
pub struct Location<'src> {
    pub line: usize,
    pub column: usize,
    pub char_idx: usize,
    pub byte_idx: usize,
    pub path: &'src str,
}

impl<'src> Location<'src> {
    pub fn new(path: &'src str) -> Self {
        Self {
            line: 1,
            column: 1,
            char_idx: 0,
            byte_idx: 0,
            path,
        }
    }
}

impl<'src> Location<'src> {
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

    pub fn until(self, end: Location<'src>) -> Span {
        Span { start: self, end }
    }
}
