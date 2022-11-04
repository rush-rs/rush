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
        params: [$(($param:tt, $type:tt)),* $(,)?],
        return_type: $return_type:tt,
        block: $block:tt $(,)?
    )) => {
        FunctionDefinition {
            span: span!($start..$end),
            name: tree!($ident),
            params: vec![$((tree!($param), tree!($type))),*],
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
    ((Spanned @ $start:literal .. $end:literal, $value:expr)) => {
        Spanned {
            span: span!($start..$end),
            inner: $value,
        }
    };
    ((Int $($rest:tt)*)) => { Expression::Int(tree!((Spanned $($rest)*))) };
    ((Float $($rest:tt)*)) => { Expression::Float(tree!((Spanned $($rest)*))) };
    ((Bool $($rest:tt)*)) => { Expression::Bool(tree!((Spanned $($rest)*))) };
    ((Ident $($rest:tt)*)) => { Expression::Ident(tree!((Spanned $($rest)*))) };
    ((Grouped @ $start:literal .. $end:literal, $expr:tt)) => {
        Expression::Grouped(Spanned {
            span: span!($start..$end),
            inner: tree!($expr).into(),
        })
    };
    ((
        PrefixExpr @ $start:literal .. $end:literal,
        op: $op:ident,
        expr: $expr:tt $(,)?
    )) => {
        Expression::Prefix(PrefixExpr {
            span: span!($start..$end),
            op: PrefixOp::$op,
            expr: tree!($expr),
        }.into())
    };
    ((
        InfixExpr @ $start:literal .. $end:literal,
        lhs: $lhs:tt,
        op: $op:ident,
        rhs: $rhs:tt $(,)?
    )) => {
        Expression::Infix(InfixExpr {
            span: span!($start..$end),
            lhs: tree!($lhs),
            op: InfixOp::$op,
            rhs: tree!($rhs),
        }.into())
    };
    ((
        AssignExpr @ $start:literal .. $end:literal,
        assignee: $assignee:tt,
        op: $op:ident,
        expr: $expr:tt $(,)?
    )) => {
        Expression::Assign(AssignExpr {
            span: span!($start..$end),
            assignee: tree!($assignee),
            op: AssignOp::$op,
            expr: tree!($expr),
        }.into())
    };
    ((
        CallExpr @ $start:literal .. $end:literal,
        expr: $expr:tt,
        args: [$($arg:tt),* $(,)?] $(,)?
    )) => {
        Expression::Call(CallExpr {
            span: span!($start..$end),
            expr: tree!($expr),
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
}

macro_rules! tokens {
    ($($kind:ident $(($($tt:tt)*))? @ $start:literal .. $end:literal),* $(,)?) => {
        [$(TokenKind::$kind $(($($tt)*))? .spanned(span!($start..$end))),*]
    };
}
