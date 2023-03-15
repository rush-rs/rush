use std::{
    collections::{HashSet, VecDeque},
    fmt::Display,
};

use rush_analyzer::{AssignOp, InfixOp, PrefixOp, Type};

fn display_stmts(stmts: &[Statement]) -> String {
    stmts
        .iter()
        .map(|s| format!("    {stmt}", stmt = s.to_string().replace('\n', "\n    ")))
        .collect::<Vec<String>>()
        .join("\n")
}

#[derive(Debug, Clone, Copy)]
pub enum CType {
    LongLongInt(usize),
    Bool(usize),
    Char(usize),
    Double(usize),
    Void,
}

impl Display for CType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CType::LongLongInt(ptr) => write!(f, "long long int{}", "*".repeat(*ptr)),
            CType::Bool(ptr) => write!(f, "bool{}", "*".repeat(*ptr)),
            CType::Char(ptr) => write!(f, "char{}", "*".repeat(*ptr)),
            CType::Double(ptr) => write!(f, "double{}", "*".repeat(*ptr)),
            CType::Void => write!(f, "void"),
        }
    }
}

impl From<Type> for CType {
    fn from(src: Type) -> Self {
        match src {
            Type::Int(ptr) => Self::LongLongInt(ptr),
            Type::Float(ptr) => Self::Double(ptr),
            Type::Bool(ptr) => Self::Bool(ptr),
            Type::Char(ptr) => Self::Char(ptr),
            Type::Unit | Type::Never => Self::Void,
            Type::Unknown => panic!("tried to convert unknown type to CType"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CProgram {
    pub includes: HashSet<&'static str>,
    pub globals: Vec<Statement>,
    pub functions: VecDeque<FnDefinition>,
}

impl Display for CProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let includes = match self.includes.is_empty() {
            true => String::new(),
            false => format!(
                "{}\n",
                self.includes
                    .iter()
                    .map(|s| format!("#include <{s}>\n"))
                    .collect::<String>()
            ),
        };

        let globals = match self.globals.is_empty() {
            true => String::new(),
            false => format!(
                "{}\n",
                self.globals
                    .iter()
                    .map(|g| format!("{g}\n"))
                    .collect::<String>()
            ),
        };

        let func_signatures = match self.functions.is_empty() {
            true => String::new(),
            false => format!(
                "{}\n",
                self.functions
                    .iter()
                    .map(|f| format! {"{};\n", FnSignature::from(f)})
                    .collect::<String>()
            ),
        };

        let func_definitions = match self.functions.is_empty() {
            true => String::new(),
            false => self
                .functions
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<String>>()
                .join("\n\n"),
        };

        write!(f, "{includes}{globals}{func_signatures}{func_definitions}")
    }
}

#[derive(Debug, Clone)]
pub struct FnSignature {
    pub name: String,
    pub type_: CType,
    pub params: Vec<(String, CType)>,
}

impl From<&FnDefinition> for FnSignature {
    fn from(value: &FnDefinition) -> Self {
        Self {
            name: value.name.clone(),
            type_: value.type_,
            params: value.params.clone(),
        }
    }
}

impl Display for FnSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params = self
            .params
            .iter()
            .map(|(_, type_)| type_.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        write!(
            f,
            "{type_} {name}({params})",
            type_ = self.type_,
            name = self.name,
        )
    }
}

#[derive(Debug, Clone)]
pub struct FnDefinition {
    pub name: String,
    pub type_: CType,
    pub params: Vec<(String, CType)>,
    pub body: Vec<Statement>,
}

impl Display for FnDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params = self
            .params
            .iter()
            .map(|(name, type_)| format!("{type_} {name}"))
            .collect::<Vec<String>>()
            .join(", ");

        let block = display_stmts(&self.body);
        let body = match block.contains('\n') {
            true => format!("{{\n{block}\n}}"),
            false => format!("{{ {block} }}", block = block.trim_start()),
        };

        write!(
            f,
            "{type_} {name}({params}) {body}",
            type_ = self.type_,
            name = self.name,
        )
    }
}

#[derive(Debug, Clone)]
pub enum Statement {
    Comment(&'static str),
    VarDeclaration(VarDeclaration),
    VarDefinition(String, CType),
    Return(Option<Expression>),
    While(WhileStmt),
    Break,
    Goto(String),
    Label(String),
    Expr(Expression),
    Assign(AssignStmt),
    If(IfStmt),
}

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Statement::VarDeclaration(node) => write!(f, "{node}"),
            Statement::VarDefinition(ident, type_) => write!(f, "{type_} {ident};"),
            Statement::Return(expr) => match expr {
                Some(expr) => write!(f, "return {expr};"),
                None => write!(f, "return;"),
            },
            Statement::While(node) => write!(f, "{node}"),
            Statement::Break => write!(f, "break;"),
            Statement::Goto(label) => write!(f, "goto {label};"),
            Statement::Label(label) => write!(f, "{label}:;"),
            Statement::Expr(expr) => write!(f, "{expr};"),
            Statement::Assign(node) => write!(f, "{node}"),
            Statement::Comment(msg) => write!(f, "// {msg}"),
            Statement::If(node) => write!(f, "{node}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VarDeclaration {
    pub name: String,
    pub type_: CType,
    pub expr: Expression,
}

impl Display for VarDeclaration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{type_} {ident} = {expr};",
            type_ = self.type_,
            ident = self.name,
            expr = self.expr
        )
    }
}

#[derive(Debug, Clone)]
pub struct WhileStmt {
    pub cond: Expression,
    pub body: Vec<Statement>,
}

impl Display for WhileStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "while ({cond}) {{\n{body}\n}}",
            cond = self.cond,
            body = display_stmts(&self.body)
        )
    }
}

#[derive(Debug, Clone)]
pub struct AssignStmt {
    pub assignee: String,
    pub assignee_ptr_count: usize,
    pub op: AssignOp,
    pub expr: Expression,
}

impl Display for AssignStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{ptrs}{ident} {op} {expr};",
            ptrs = "*".repeat(self.assignee_ptr_count),
            ident = self.assignee,
            op = self.op,
            expr = self.expr
        )
    }
}

#[derive(Debug, Clone)]
pub struct IfStmt {
    pub cond: Expression,
    pub then_block: Vec<Statement>,
    pub else_block: Option<Vec<Statement>>,
}

impl Display for IfStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let then_block = display_stmts(&self.then_block);

        let else_part = match &self.else_block {
            Some(stmts) => {
                format!(" else {{\n{body}\n}}", body = display_stmts(stmts))
            }
            None => String::new(),
        };

        write!(
            f,
            "if ({cond}) {{\n{then_block}\n}}{else_part}",
            cond = self.cond
        )
    }
}

#[derive(Debug, Clone)]
pub enum Expression {
    Call(Box<CallExpr>),
    Prefix(Box<PrefixExpr>),
    Infix(Box<InfixExpr>),
    Deref((usize, String)),
    Cast(Box<CastExpr>),
    Int(i64),
    Bool(bool),
    Char(u8),
    Float(f64),
    Ident(String),
    Grouped(Box<Expression>),
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Call(node) => write!(f, "{node}"),
            Expression::Prefix(node) => write!(f, "{node}"),
            Expression::Infix(node) => write!(f, "{node}"),
            Expression::Deref((count, ident)) => {
                write!(f, "{deref}{ident}", deref = "*".repeat(*count))
            }
            Expression::Cast(node) => write!(f, "{node}"),
            Expression::Int(val) => write!(f, "{val}"),
            Expression::Bool(val) => write!(f, "{val}"),
            Expression::Char(val) => write!(f, "{val}"),
            Expression::Float(val) => write!(f, "{val}"),
            Expression::Ident(ident) => write!(f, "{ident}"),
            Expression::Grouped(node) => write!(f, "({node})"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CallExpr {
    pub func: String,
    pub args: Vec<Expression>,
}

impl Display for CallExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let args = self
            .args
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        let args = match args.contains('\n') {
            true => format!("(\n    {args}\n)", args = args.replace('\n', "\n    ")),
            false => format!("({args})"),
        };

        write!(f, "{func}{args}", func = self.func)
    }
}

#[derive(Debug, Clone)]
pub struct PrefixExpr {
    pub expr: Expression,
    pub op: PrefixOp,
}

impl Display for PrefixExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{op}{expr}", op = self.op, expr = self.expr)
    }
}

#[derive(Debug, Clone)]
pub struct InfixExpr {
    pub lhs: Expression,
    pub rhs: Expression,
    pub op: InfixOp,
}

impl Display for InfixExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lhs = self.lhs.to_string();
        let rhs = self.rhs.to_string();

        let split_here = lhs.split('\n').last().unwrap().len() > 80;

        match split_here {
            true => write!(f, "{lhs}\n    {op} {rhs}", op = self.op),
            false => write!(f, "{lhs} {op} {rhs}", op = self.op),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CastExpr {
    pub expr: Expression,
    pub type_: CType,
}

impl Display for CastExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({type_}) {expr}", type_ = self.type_, expr = self.expr)
    }
}
