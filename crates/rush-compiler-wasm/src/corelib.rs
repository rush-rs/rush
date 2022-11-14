use crate::{instructions, types, utils::Leb128, Compiler};

impl Compiler<'_> {
    fn call_internal(
        &mut self,
        name: &'static str,
        signature: Vec<u8>,
        locals: Vec<u8>,
        local_names: &[&str],
        body: &[u8],
    ) {
        let idx = match self.builtin_functions.get(name) {
            Some(idx) => idx,
            None => {
                let type_idx = self.type_section.len().to_uleb128();

                // add signature to type section
                self.type_section.push(signature);

                // add to function section
                let func_idx = (self.function_section.len() + self.import_count).to_uleb128();
                self.function_section.push(type_idx);

                // save in builtin_functions map
                self.builtin_functions.insert(name, func_idx);
                let func_idx = &self.builtin_functions[name];

                // add to end of code section
                self.builtins_code
                    .push([&(body.len() + locals.len()).to_uleb128(), &locals, body].concat());

                // add name to name section
                self.function_names.push(
                    [
                        &func_idx[..],            // function index
                        &name.len().to_uleb128(), // string len
                        name.as_bytes(),          // name
                    ]
                    .concat(),
                );

                // add local names to name section
                self.local_names.push((
                    func_idx.clone(),
                    local_names
                        .iter()
                        .enumerate()
                        .map(|(idx, name)| {
                            [
                                &idx.to_uleb128()[..],    // local index
                                &name.len().to_uleb128(), // string len
                                name.as_bytes(),          // local name
                            ]
                            .concat()
                        })
                        .collect(),
                ));

                func_idx
            }
        };

        // push call instruction
        self.function_body.push(instructions::CALL);
        self.function_body.extend_from_slice(idx);
    }

    pub(crate) fn __rush_internal_cast_int_to_char(&mut self) {
        self.call_internal(
            "__rush_internal_cast_int_to_char",
            vec![
                types::FUNC,
                1, // num of params
                types::I64,
                1, // num of return vals
                types::I32,
            ],
            vec![0],  // no locals
            &["int"], // name of param
            &[
                // get param
                instructions::LOCAL_GET,
                0,
                // if > 0x7F
                instructions::I64_CONST,
                0xFF, // 0x7F in signed LEB128 is [0xFF, 0x00]
                0x00,
                instructions::I64_GT_S,
                instructions::IF,
                types::I32,
                // then return 0x7F
                instructions::I32_CONST,
                0xFF, // 0x7F in signed LEB128 is [0xFF, 0x00]
                0x00,
                // else if < 0x00
                instructions::ELSE,
                instructions::LOCAL_GET,
                0,
                instructions::I64_CONST,
                0,
                instructions::I64_LT_S,
                instructions::IF,
                types::I32,
                // then return 0x00
                instructions::I32_CONST,
                0x00,
                // else convert to i32
                instructions::ELSE,
                instructions::LOCAL_GET,
                0,
                instructions::I32_WRAP_I64,
                // end if
                instructions::END,
                instructions::END,
                // end function body
                instructions::END,
            ],
        );
    }

    pub(crate) fn __rush_internal_cast_float_to_char(&mut self) {
        self.call_internal(
            "__rush_internal_cast_float_to_char",
            vec![
                types::FUNC,
                1, // num of params
                types::F64,
                1, // num of return vals
                types::I32,
            ],
            vec![1, 1, types::I32],
            &["float", "tmp"], // names of params and locals
            &[
                // get param
                instructions::LOCAL_GET,
                0,
                // convert to i32
                instructions::I32_TRUNC_SAT_F64_U[0],
                instructions::I32_TRUNC_SAT_F64_U[1],
                // set local to result
                instructions::LOCAL_TEE,
                1,
                // if > 0x7F
                instructions::I32_CONST,
                0xFF, // 0x7F in signed LEB128 is [0xFF, 0x00]
                0x00,
                instructions::I32_GT_U,
                instructions::IF,
                types::I32,
                // then return 0x7F
                instructions::I32_CONST,
                0xFF, // 0x7F in signed LEB128 is [0xFF, 0x00]
                0x00,
                // else use value
                instructions::ELSE,
                instructions::LOCAL_GET,
                1,
                // end if
                instructions::END,
                // end function body
                instructions::END,
            ],
        );
    }

    pub(crate) fn __rush_internal_pow_int(&mut self) {
        self.call_internal(
            "__rush_internal_pow_int",
            vec![
                types::FUNC,
                2,          // num of params
                types::I64, // base
                types::I64, // exponent
                1,          // num of return vals
                types::I64,
            ],
            vec![1, 1, types::I64],               // 1 local i64: accumulator
            &["base", "exponent", "accumulator"], // names of params and locals
            &[
                // if exponent < 0
                instructions::LOCAL_GET, // get
                1,                       // exponent
                instructions::I64_CONST,
                0,
                instructions::I64_LT_S,
                instructions::IF, // if
                types::I64,       // with result `int`
                // then return 0
                instructions::I64_CONST,
                0,
                // else calculate with loop
                instructions::ELSE,
                // -- set accumulator to 1
                instructions::I64_CONST,
                1,
                instructions::LOCAL_SET,
                2,
                // -- begin block around loop
                instructions::BLOCK,
                types::VOID, // with result `()`
                // -- begin loop
                instructions::LOOP,
                types::VOID, // with result `()`
                // -- break if exponent == 0
                instructions::LOCAL_GET, // get
                1,                       // exponent
                instructions::I64_EQZ,   // == 0
                instructions::BR_IF,     // conditional jump
                1,                       // to end of outer block
                // -- decrement exponent
                instructions::LOCAL_GET, // get
                1,                       // exponent
                instructions::I64_CONST,
                1,
                instructions::I64_SUB,   // subtract 1
                instructions::LOCAL_SET, // set
                1,                       // exponent
                // -- multiply accumulator with base
                instructions::LOCAL_GET, // get
                2,                       // accumulator
                instructions::LOCAL_GET, // get
                0,                       // base
                instructions::I64_MUL,   // multiply
                instructions::LOCAL_SET, // set
                2,                       // accumulator
                // -- jump to start of loop
                instructions::BR,
                0,
                // -- end loop
                instructions::END,
                // -- end block
                instructions::END,
                // -- return accumulator
                instructions::LOCAL_GET,
                2,
                // end if
                instructions::END,
                // end function body
                instructions::END,
            ],
        );
    }

    /////////////////////////

    fn call_wasi(&mut self, name: &'static str, wasi_name: &str, signature: Vec<u8>) {
        let idx = match self.builtin_functions.get(name) {
            Some(idx) => idx,
            None => {
                let type_idx = self.type_section.len().to_uleb128();

                // add signature to type section
                self.type_section.push(signature);

                // save in builtin_functions map
                let func_idx = self.import_section.len().to_uleb128();
                self.builtin_functions.insert(name, func_idx);
                let func_idx = &self.builtin_functions[name];

                // add import from WASI
                self.import_section.push(
                    [
                        &[22][..],                     // module string len
                        b"wasi_snapshot_preview1",     // module name
                        &wasi_name.len().to_uleb128(), // func name string len
                        wasi_name.as_bytes(),          // func name
                        &[0],                          // import of type `func`
                        &type_idx,                     // index of func signature in type section
                    ]
                    .concat(),
                );

                // add name to name section
                self.imported_function_names.push(
                    [
                        &func_idx[..],            // function index
                        &name.len().to_uleb128(), // string len
                        name.as_bytes(),          // name
                    ]
                    .concat(),
                );

                func_idx
            }
        };

        // push call instruction
        self.function_body.push(instructions::CALL);
        self.function_body.extend_from_slice(idx);
    }

    pub(crate) fn __wasi_exit(&mut self) {
        self.function_body.push(instructions::I32_WRAP_I64);
        self.call_wasi(
            "__wasi_exit",
            "proc_exit",
            vec![
                types::FUNC,
                1, // num of params
                types::I32,
                0, // num of return vals
            ],
        );
    }
}
