use std::mem;

use rush_analyzer::ast::*;

mod instructions;
mod types;

#[derive(Debug, Default)]
pub struct Compiler {
    function_body: Vec<u8>,
    locals: Vec<Vec<u8>>,

    type_section: Vec<Vec<u8>>,     // 1
    import_section: Vec<Vec<u8>>,   // 2
    function_section: Vec<Vec<u8>>, // 3
    table_section: Vec<Vec<u8>>,    // 4
    memory_section: Vec<Vec<u8>>,   // 5
    global_section: Vec<Vec<u8>>,   // 6
    export_section: Vec<Vec<u8>>,   // 7
    start_section: Vec<u8>,         // 8
    element_section: Vec<Vec<u8>>,  // 9
    code_section: Vec<Vec<u8>>,     // 10
    data_section: Vec<Vec<u8>>,     // 11
    data_count_section: Vec<u8>,    // 12
}

impl<'src> Compiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compile(mut self, tree: AnalyzedProgram<'src>) -> Vec<u8> {
        self.program(tree);
        [
            &b"\0asm"[..],        // magic
            &1_i32.to_le_bytes(), // spec version 1
            &Self::section(1, self.type_section),
            &Self::section(2, self.import_section),
            &Self::section(3, self.function_section),
            &Self::section(4, self.table_section),
            &Self::section(5, self.memory_section),
            &Self::section(6, self.global_section),
            &Self::section(7, self.export_section),
            // TODO: start and data_count section
            // &Self::section(8, self.start_section),
            &Self::section(9, self.element_section),
            &Self::section(10, self.code_section),
            &Self::section(11, self.data_section),
            // &Self::section(12, self.data_count_section),
        ]
        .concat()
    }

    fn section(id: u8, section: Vec<Vec<u8>>) -> Vec<u8> {
        if section.is_empty() {
            return vec![];
        }

        // start new buf with section id
        let mut buf = vec![id];

        // add vector
        buf.append(&mut Self::vector(section, true));

        // return
        buf
    }

    fn vector(vec: Vec<Vec<u8>>, add_byte_count: bool) -> Vec<u8> {
        let mut buf = vec![];

        // combine vector contents
        let mut combined_bytes = vec.concat();

        // get length of vector
        let mut vector_len = to_uleb128(vec.len() as u64);

        // add byte count
        if add_byte_count {
            buf.append(&mut to_uleb128(
                (combined_bytes.len() + vector_len.len()) as u64,
            ));
        }

        // add vector length
        buf.append(&mut vector_len);

        // add contents
        buf.append(&mut combined_bytes);

        // return
        buf
    }

    /////////////////////////

    fn program(&mut self, node: AnalyzedProgram) {
        self.main_fn(node.main_fn);
        for func in node.functions {
            self.function_definition(func);
        }
    }

    fn main_fn(&mut self, body: AnalyzedBlock) {
        self.type_section.push(vec![
            types::FUNC,
            0, // num of params
            0, // num of return vals
        ]);

        // index of type in type_section (main func is always 0)
        self.function_section.push(vec![0]);

        // export main func as WASI `_start` func
        self.export_section.push(
            [
                &[
                    6, // string len
                ][..],
                b"_start", // name of export
                &[
                    0, // export kind (0 = func)
                    0, // index of func in function_section (main func is always 0)
                ],
            ]
            .concat(),
        );

        // function body
        self.function_body(body);
    }

    fn function_definition(&mut self, node: AnalyzedFunctionDefinition) {
        todo!()
    }

    fn function_body(&mut self, node: AnalyzedBlock) {
        // TODO: body
        self.function_body.push(instructions::END);

        // take self.locals
        let locals = Self::vector(mem::take(&mut self.locals), false);

        // push function
        self.code_section.push(
            [
                // body size
                to_uleb128((self.function_body.len() + locals.len()) as u64),
                // locals
                locals,
                // function budy
                mem::take(&mut self.function_body),
            ]
            .concat(),
        );
    }
}

fn to_uleb128(num: u64) -> Vec<u8> {
    if num < 128 {
        return vec![num as u8];
    }

    let mut buf = vec![];
    leb128::write::unsigned(&mut buf, num).expect("writing to a Vec should never fail");
    buf
}
