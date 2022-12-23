use rush_analyzer::{
    ast::{
        AnalyzedAssignExpr, AnalyzedBlock, AnalyzedCallExpr, AnalyzedExpression, AnalyzedForStmt,
        AnalyzedFunctionDefinition, AnalyzedInfixExpr, AnalyzedLetStmt, AnalyzedLoopStmt,
        AnalyzedProgram, AnalyzedReturnStmt, AnalyzedStatement, AnalyzedWhileStmt,
    },
    Type,
};

pub struct Transpiler {
    output: Vec<String>,
    curr_fragment: usize,
    indent: usize,
}

impl<'src> Transpiler {
    pub fn new() -> Self {
        Self {
            output: vec![String::new()],
            curr_fragment: 0,
            indent: 0,
        }
    }

    #[inline]
    fn insert(&mut self, text: &str) {
        self.output[self.curr_fragment].push_str(text);
    }

    fn insert_ln(&mut self, text: &str) {
        let padding = " ".repeat(self.indent);
        self.insert(&padding);
        self.insert(text);
        self.output[self.curr_fragment].push('\n');
    }

    pub fn transpile(mut self, tree: AnalyzedProgram<'src>) -> String {
        self.insert_ln("#include <stdbool.h>");
        self.insert_ln("#include <stdlib.h>");
        self.insert_ln("");

        for func in tree.functions.into_iter().filter(|func| func.used) {
            self.fn_declaration(func);
            self.output[self.curr_fragment].push_str("\n\n")
        }

        self.main_fn(tree.main_fn);

        self.output.into_iter().collect()
    }

    fn fn_declaration(&mut self, node: AnalyzedFunctionDefinition<'src>) {
        let type_ = display_type(node.return_type);

        let params = node
            .params
            .into_iter()
            .map(|p| {
                format!(
                    "{type_} {name}",
                    type_ = display_type(p.type_),
                    name = p.name
                )
            })
            .collect::<Vec<String>>()
            .join(", ");

        let body = self.block(node.block);

        self.insert(&format!(
            "{type_} {name}({params}) {body}",
            name = node.name
        ));
    }

    fn main_fn(&mut self, node: AnalyzedBlock<'src>) {
        let block = self.block(node);
        self.insert(&format!("int main() {block}"));
    }

    fn block(&mut self, node: AnalyzedBlock<'src>) -> String {
        self.indent += 4;

        let inner_indent = " ".repeat(self.indent);

        let stmts = node
            .stmts
            .into_iter()
            .map(|s| self.statement(s))
            .collect::<Vec<String>>()
            .join(&format!("\n{}", inner_indent));

        // TODO: do special things if block has expression at the end
        let expr = match node.expr {
            Some(expr) => format!("\n{inner_indent}{};\n", self.expression(expr)),
            None => "\n".to_string(),
        };

        self.indent -= 4;

        let outer_indent = " ".repeat(self.indent);

        format!("{{\n{inner_indent}{stmts}{expr}{outer_indent}}}")
    }

    fn statement(&mut self, node: AnalyzedStatement<'src>) -> String {
        let text = match node {
            AnalyzedStatement::Let(node) => self.let_stmt(node),
            AnalyzedStatement::Return(node) => self.return_stmt(node),
            AnalyzedStatement::Loop(node) => self.loop_stmt(node),
            AnalyzedStatement::While(node) => self.while_stmt(node),
            AnalyzedStatement::For(node) => self.for_stmt(node),
            AnalyzedStatement::Break => "break".to_string(),
            AnalyzedStatement::Continue => "continue".to_string(),
            AnalyzedStatement::Expr(node) => self.expression(node),
        };
        format!("{text};")
    }

    fn let_stmt(&mut self, node: AnalyzedLetStmt<'src>) -> String {
        let type_ = display_type(node.expr.result_type());

        let expr = self.expression(node.expr);

        format!("{type_} {} = {expr}", node.name)
    }

    fn return_stmt(&mut self, node: AnalyzedReturnStmt<'src>) -> String {
        match node {
            Some(expr) => format!("return {}", self.expression(expr)),
            None => "return".to_string(),
        }
    }

    fn loop_stmt(&mut self, node: AnalyzedLoopStmt<'src>) -> String {
        let block = self.block(node.block);
        format!("while (true) {block}")
    }

    fn while_stmt(&mut self, node: AnalyzedWhileStmt<'src>) -> String {
        let cond = self.expression(node.cond);
        let block = self.block(node.block);
        format!("while ({cond}) {block}")
    }

    fn for_stmt(&mut self, node: AnalyzedForStmt<'src>) -> String {
        let init = format!(
            "{type_} {ident} = {expr}",
            type_ = display_type(node.initializer.result_type()),
            ident = node.ident,
            expr = self.expression(node.initializer)
        );
        let cond = self.expression(node.cond);
        let update = self.expression(node.update);
        let block = self.block(node.block);
        format!("for ({init}; {cond}; {update}) {block}")
    }

    fn expression(&mut self, node: AnalyzedExpression<'src>) -> String {
        let text = match node {
            AnalyzedExpression::Block(_) => todo!(),
            AnalyzedExpression::If(_) => todo!(),
            AnalyzedExpression::Int(val) => format!("{val}"),
            AnalyzedExpression::Float(val) => format!("{val}"),
            AnalyzedExpression::Bool(val) => format!("{val}"),
            AnalyzedExpression::Char(val) => format!("'{val}'"),
            AnalyzedExpression::Ident(ident) => ident.ident.to_string(),
            AnalyzedExpression::Prefix(_) => todo!(),
            AnalyzedExpression::Infix(node) => self.infix_expr(*node),
            AnalyzedExpression::Assign(node) => self.assign_expr(*node),
            AnalyzedExpression::Call(node) => self.call_expr(*node),
            AnalyzedExpression::Cast(_) => todo!(),
            AnalyzedExpression::Grouped(expr) => format!("({expr})", expr = self.expression(*expr)),
        };

        text
    }

    fn infix_expr(&mut self, node: AnalyzedInfixExpr<'src>) -> String {
        let lhs = self.expression(node.lhs);
        let rhs = self.expression(node.rhs);
        format!("{lhs} {op} {rhs}", op = node.op)
    }

    fn assign_expr(&mut self, node: AnalyzedAssignExpr<'src>) -> String {
        format!(
            "{ident} {op} {rhs}",
            ident = node.assignee,
            op = node.op,
            rhs = self.expression(node.expr)
        )
    }

    fn call_expr(&mut self, node: AnalyzedCallExpr<'src>) -> String {
        let args = node
            .args
            .into_iter()
            .map(|expr| self.expression(expr))
            .collect::<Vec<String>>()
            .join(", ");

        format!("{}({args})", node.func)
    }
}

fn display_type(type_: Type) -> String {
    match type_ {
        Type::Int(ptr) => format!("int{}", "*".repeat(ptr)),
        Type::Float(ptr) => format!("float{}", "*".repeat(ptr)),
        Type::Char(ptr) => format!("char{}", "*".repeat(ptr)),
        Type::Bool(ptr) => format!("bool{}", "*".repeat(ptr)),
        Type::Unit => todo!(),
        Type::Never => todo!(),
        Type::Unknown => todo!(),
    }
}
