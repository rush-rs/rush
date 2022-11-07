#[macro_export]
macro_rules! tree {
    ((None)) => { None };
    ((Some($($node:tt)*))) => { Some(tree!(($($node)*))) };

    ((Program @ $start:literal .. $end:literal, [$($func:tt),* $(,)?])) => {
        Program {
            span: span!($start..$end),
            functions: vec![$(tree!($func)),*],
        }
    };
    ((
        FunctionDefinition @ $start:literal .. $end:literal,
        name: $ident:tt,
        params @ $params_start:literal .. $params_end:literal: [$(($param:tt, $type:tt)),* $(,)?],
        return_type: $return_type:tt,
        block: $block:tt $(,)?
    )) => {
        FunctionDefinition {
            span: span!($start..$end),
            name: tree!($ident),
            params: tree!((vec![$((tree!($param), tree!($type))),*], @ $params_start..$params_end)),
            return_type: tree!($return_type),
            block: tree!($block),
        }
    };

    ((
        LetStmt @ $start:literal .. $end:literal,
        mutable: $mut:literal,
        name: $ident:tt,
        type: $type:tt,
        expr: $expr:tt $(,)?
    )) => {
        Statement::Let(LetStmt {
            span: span!($start..$end),
            mutable: $mut,
            name: tree!($ident),
            type_: tree!($type),
            expr: tree!($expr),
        })
    };
    ((ReturnStmt @ $start:literal .. $end:literal, $expr:tt $(,)?)) => {
        Statement::Return(ReturnStmt {
            span: span!($start..$end),
            expr: tree!($expr),
        })
    };
    ((ExprStmt @ $start:literal .. $end:literal, $expr:tt $(,)?)) => {
        Statement::Expr(ExprStmt {
            span: span!($start..$end),
            expr: tree!($expr),
        })
    };

    ((
        Block @ $start:literal .. $end:literal,
        stmts: [$($stmt:tt),* $(,)?],
        expr: $expr:tt $(,)?
    )) => {
        Block {
            span: span!($start..$end),
            stmts: vec![$(tree!($stmt)),*],
            expr: tree!($expr),
        }
    };
    ((BlockExpr $($rest:tt)*)) => {
        Expression::Block(tree!((Block $($rest)*)).into())
    };
    ((
        IfExpr @ $start:literal .. $end:literal,
        cond: $cond:tt,
        then_block: $then_block:tt,
        else_block: $else_block:tt $(,)?
    )) => {
        Expression::If(IfExpr {
            span: span!($start..$end),
            cond: tree!($cond),
            then_block: tree!($then_block),
            else_block: tree!($else_block),
        }.into())
    };
    ((Int $($rest:tt)*)) => { Expression::Int(tree!(($($rest)*))) };
    ((Float $($rest:tt)*)) => { Expression::Float(tree!(($($rest)*))) };
    ((Bool $($rest:tt)*)) => { Expression::Bool(tree!(($($rest)*))) };
    ((Char $($rest:tt)*)) => { Expression::Char(tree!(($($rest)*))) };
    ((Ident $($rest:tt)*)) => { Expression::Ident(tree!(($($rest)*))) };
    ((Grouped @ $start:literal .. $end:literal, $expr:tt)) => {
        Expression::Grouped(Spanned {
            span: span!($start..$end),
            inner: tree!($expr).into(),
        })
    };
    ((
        PrefixExpr @ $start:literal .. $end:literal,
        op: $op:expr,
        expr: $expr:tt $(,)?
    )) => {
        Expression::Prefix(PrefixExpr {
            span: span!($start..$end),
            op: $op,
            expr: tree!($expr),
        }.into())
    };
    ((
        InfixExpr @ $start:literal .. $end:literal,
        lhs: $lhs:tt,
        op: $op:expr,
        rhs: $rhs:tt $(,)?
    )) => {
        Expression::Infix(InfixExpr {
            span: span!($start..$end),
            lhs: tree!($lhs),
            op: $op,
            rhs: tree!($rhs),
        }.into())
    };
    ((
        AssignExpr @ $start:literal .. $end:literal,
        assignee: $assignee:tt,
        op: $op:expr,
        expr: $expr:tt $(,)?
    )) => {
        Expression::Assign(AssignExpr {
            span: span!($start..$end),
            assignee: tree!($assignee),
            op: $op,
            expr: tree!($expr),
        }.into())
    };
    ((
        CallExpr @ $start:literal .. $end:literal,
        func: $func:tt,
        args: [$($arg:tt),* $(,)?] $(,)?
    )) => {
        Expression::Call(CallExpr {
            span: span!($start..$end),
            func: tree!($func),
            args: vec![$(tree!($arg)),*],
        }.into())
    };
    ((
        CastExpr @ $start:literal .. $end:literal,
        expr: $expr:tt,
        type: $type:expr $(,)?
    )) => {
        Expression::Cast(CastExpr {
            span: span!($start..$end),
            expr: tree!($expr),
            type_: $type,
        }.into())
    };

    (($value:expr, @ $start:literal .. $end:literal)) => {
        Spanned {
            span: span!($start..$end),
            inner: $value,
        }
    };
}

#[macro_export]
macro_rules! tokens {
    ($($kind:ident $(($($tt:tt)*))? @ $start:literal .. $end:literal),* $(,)?) => {
        [$(TokenKind::$kind $(($($tt)*))? .spanned(span!($start..$end))),*]
    };
}
