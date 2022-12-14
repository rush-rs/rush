#[macro_export]
macro_rules! analyzed_tree {
    ((None)) => { None };
    ((Some($($node:tt)*))) => { Some(analyzed_tree!(($($node)*))) };

    ((
        Program,
        globals: [$($global:tt),* $(,)?],
        functions: [$($func:tt),* $(,)?],
        main_fn: $main_fn:tt,
        used_builtins: [$($name:expr),* $(,)?] $(,)?
    )) => {
        AnalyzedProgram {
            globals: vec![$(analyzed_tree!($global)),*],
            functions: vec![$(analyzed_tree!($func)),*],
            main_fn: analyzed_tree!($main_fn),
            used_builtins: HashSet::from([$($name),*]),
        }
    };
    ((
        FunctionDefinition,
        used: $used:expr,
        name: $name:expr,
        params: [$($param:tt),* $(,)?],
        return_type: $return_type:expr,
        block: $block:tt $(,)?
    )) => {
        AnalyzedFunctionDefinition {
            used: $used,
            name: $name,
            params: vec![$(analyzed_tree!($param)),*],
            return_type: $return_type,
            block: analyzed_tree!($block),
        }
    };
    ((
        Parameter,
        mutable: $mutable:expr,
        name: $name:expr,
        type: $type:expr $(,)?
    )) => {
        AnalyzedParameter {
            mutable: $mutable,
            name: $name,
            type_: $type,
        }
    };

    ((
        Let,
        name: $name:expr,
        expr: $expr:tt $(,)?
    )) => {
        AnalyzedLetStmt {
            name: $name,
            expr: analyzed_tree!($expr),
        }
    };
    ((LetStmt $($rest:tt)*)) => {
        AnalyzedStatement::Let(analyzed_tree!((Let $($rest)*)))
    };
    ((ReturnStmt, $expr:tt $(,)?)) => {
        AnalyzedStatement::Return(analyzed_tree!($expr))
    };
    ((ExprStmt, $expr:tt $(,)?)) => {
        AnalyzedStatement::Expr(analyzed_tree!($expr))
    };

    ((
        Block -> $result_type:expr,
        stmts: [$($stmt:tt),* $(,)?],
        expr: $expr:tt $(,)?
    )) => {
        AnalyzedBlock {
            result_type: $result_type,
            stmts: vec![$(analyzed_tree!($stmt)),*],
            expr: analyzed_tree!($expr),
        }
    };
    ((BlockExpr $($rest:tt)*)) => {
        AnalyzedExpression::Block(analyzed_tree!((Block $($rest)*)).into())
    };
    ((
        IfExpr -> $result_type:expr,
        cond: $cond:tt,
        then_block: $then_block:tt,
        else_block: $else_block:tt $(,)?
    )) => {
        AnalyzedExpression::If(AnalyzedIfExpr {
            result_type: $result_type,
            cond: analyzed_tree!($cond),
            then_block: analyzed_tree!($then_block),
            else_block: analyzed_tree!($else_block),
        }.into())
    };
    ((Int $expr:expr)) => { AnalyzedExpression::Int($expr) };
    ((Float $expr:expr)) => { AnalyzedExpression::Float($expr) };
    ((Bool $expr:expr)) => { AnalyzedExpression::Bool($expr) };
    ((Char $expr:expr)) => { AnalyzedExpression::Char($expr) };
    ((Ident -> $type:expr, $ident:expr)) => {
        AnalyzedExpression::Ident(AnalyzedIdentExpr {
            result_type: $type,
            ident: $ident,
        })
    };
    ((Grouped $expr:tt)) => {
        AnalyzedExpression::Grouped(analyzed_tree!($expr).into())
    };
    ((
        PrefixExpr -> $result_type:expr,
        op: $op:expr,
        expr: $expr:tt $(,)?
    )) => {
        AnalyzedExpression::Prefix(AnalyzedPrefixExpr {
            result_type: $result_type,
            op: $op,
            expr: analyzed_tree!($expr),
        }.into())
    };
    ((
        InfixExpr -> $result_type:expr,
        lhs: $lhs:tt,
        op: $op:expr,
        rhs: $rhs:tt $(,)?
    )) => {
        AnalyzedExpression::Infix(AnalyzedInfixExpr {
            result_type: $result_type,
            lhs: analyzed_tree!($lhs),
            op: $op,
            rhs: analyzed_tree!($rhs),
        }.into())
    };
    ((
        AssignExpr -> $result_type:expr,
        assignee: $assignee:expr,
        op: $op:expr,
        expr: $expr:tt $(,)?
    )) => {
        AnalyzedExpression::Assign(AnalyzedAssignExpr {
            result_type: $result_type,
            assignee: $assignee,
            op: $op,
            expr: analyzed_tree!($expr),
        }.into())
    };
    ((
        CallExpr -> $result_type:expr,
        func: $func:expr,
        args: [$($arg:tt),* $(,)?] $(,)?
    )) => {
        AnalyzedExpression::Call(AnalyzedCallExpr {
            result_type: $result_type,
            func: $func,
            args: vec![$(analyzed_tree!($arg)),*],
        }.into())
    };
    ((
        CastExpr -> $result_type:expr,
        expr: $expr:tt,
        type: $type:expr $(,)?
    )) => {
        AnalyzedExpression::Cast(AnalyzedCastExpr {
            result_type: $result_type,
            expr: analyzed_tree!($expr),
            type_: $type,
        }.into())
    };
}
