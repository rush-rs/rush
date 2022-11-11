use crate::{instructions, types, utils::Leb128, Compiler};

fn call_builtin(
    c: &mut Compiler,
    name: &'static str,
    signature: Vec<u8>,
    locals: Vec<u8>,
    local_names: &[&str],
    body: &[u8],
) {
    let idx = match c.builtin_functions.get(name) {
        Some(idx) => idx,
        None => {
            let type_idx = c.type_section.len().to_uleb128();

            // add signature to type section
            c.type_section.push(signature);

            // add to function section
            let func_idx = (c.function_section.len() + c.import_count).to_uleb128();
            c.function_section.push(type_idx);

            // save in builtin_functions map
            c.builtin_functions.insert(name, func_idx);
            let func_idx = &c.builtin_functions[name];

            // add to end of code section
            c.builtins_code
                .push([&(body.len() + locals.len()).to_uleb128(), &locals, body].concat());

            // add name to name section
            c.function_names.push(
                [
                    &func_idx[..],            // function index
                    &name.len().to_uleb128(), // string len
                    name.as_bytes(),          // name
                ]
                .concat(),
            );

            // add local names to name section
            c.local_names.push((
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
    c.function_body.push(instructions::CALL);
    c.function_body.extend_from_slice(idx);
}

pub fn cast_int_to_char(c: &mut Compiler) {
    call_builtin(
        c,
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

pub fn cast_float_to_char(c: &mut Compiler) {
    call_builtin(
        c,
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

pub fn pow_int(c: &mut Compiler) {
    call_builtin(
        c,
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
            instructions::LOCAL_GET,
            1,
            instructions::I64_CONST,
            0,
            instructions::I64_LT_S,
            instructions::IF,
            types::I64,
            // then return 0
            instructions::I64_CONST,
            0,
            // else if exponent == 0
            instructions::ELSE,
            instructions::LOCAL_GET,
            1,
            instructions::I64_EQZ,
            instructions::IF,
            types::I64,
            // then return 1
            instructions::I64_CONST,
            1,
            // else calculate with loop
            instructions::ELSE,
            // -- set accumulator to base
            instructions::LOCAL_GET,
            0,
            instructions::LOCAL_SET,
            2,
            // -- begin loop
            instructions::LOOP,
            types::VOID,
            // -- begin block
            instructions::BLOCK,
            types::VOID,
            // -- subtract 1 from exponent
            instructions::LOCAL_GET,
            1,
            instructions::I64_CONST,
            1,
            instructions::I64_SUB,
            instructions::LOCAL_TEE,
            1,
            // -- break if exponent is 0
            instructions::I64_EQZ,
            instructions::BR_IF,
            0, // branch depth, 0 = end of block
            // -- multiply accumulator with base
            instructions::LOCAL_GET,
            2,
            instructions::LOCAL_GET,
            0,
            instructions::I64_MUL,
            instructions::LOCAL_SET,
            2,
            // -- continue loop
            instructions::BR,
            1, // branch depth, 1 = start of loop
            // -- end block
            instructions::END,
            // -- end loop
            instructions::END,
            // -- get result in accumulator
            instructions::LOCAL_GET,
            2,
            // end if
            instructions::END,
            instructions::END,
            // end function body
            instructions::END,
        ],
    );
}
