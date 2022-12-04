use std::fmt::{self, Display, Formatter};

use rush_analyzer::{
    ast::{AnalyzedExpression, AnalyzedPrefixExpr},
    InfixOp, PrefixOp, Type,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    Above,
    AboveOrEqual,
    Below,
    BelowOrEqual,

    Equal,
    NotEqual,

    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,

    Parity,
    NotParity,
}

impl Condition {
    pub fn try_from_op(op: InfixOp, signed: bool) -> Option<Self> {
        match op {
            InfixOp::Eq => Some(Self::Equal),
            InfixOp::Neq => Some(Self::NotEqual),
            InfixOp::Lt if signed => Some(Self::Less),
            InfixOp::Gt if signed => Some(Self::Greater),
            InfixOp::Lte if signed => Some(Self::LessOrEqual),
            InfixOp::Gte if signed => Some(Self::GreaterOrEqual),
            InfixOp::Lt => Some(Self::Below),
            InfixOp::Gt => Some(Self::Above),
            InfixOp::Lte => Some(Self::BelowOrEqual),
            InfixOp::Gte => Some(Self::AboveOrEqual),
            _ => None,
        }
    }

    pub fn try_from_expr(
        mut expr: AnalyzedExpression<'_>,
    ) -> Result<(Self, AnalyzedExpression<'_>, AnalyzedExpression<'_>), AnalyzedExpression<'_>>
    {
        let mut negate = false;
        loop {
            match expr {
                AnalyzedExpression::Prefix(prefix_expr) if prefix_expr.op == PrefixOp::Not => {
                    expr = prefix_expr.expr;
                    negate ^= true;
                }
                AnalyzedExpression::Infix(infix_expr) => {
                    return match Self::try_from_op(
                        infix_expr.op,
                        infix_expr.lhs.result_type() == Type::Int,
                    ) {
                        Some(cond) => Ok((
                            if negate { cond.negated() } else { cond },
                            infix_expr.lhs,
                            infix_expr.rhs,
                        )),
                        None => Err(AnalyzedExpression::Infix(infix_expr)),
                    };
                }
                expr => {
                    return Err(match negate {
                        true => AnalyzedExpression::Prefix(
                            AnalyzedPrefixExpr {
                                result_type: Type::Bool,
                                op: PrefixOp::Not,
                                expr,
                            }
                            .into(),
                        ),
                        false => expr,
                    })
                }
            }
        }
    }

    pub fn negated(self) -> Self {
        match self {
            Self::Above => Self::BelowOrEqual,
            Self::AboveOrEqual => Self::Below,
            Self::Below => Self::AboveOrEqual,
            Self::BelowOrEqual => Self::Above,
            Self::Equal => Self::NotEqual,
            Self::NotEqual => Self::Equal,
            Self::Greater => Self::LessOrEqual,
            Self::GreaterOrEqual => Self::Less,
            Self::Less => Self::GreaterOrEqual,
            Self::LessOrEqual => Self::Greater,
            Self::Parity => Self::NotParity,
            Self::NotParity => Self::Parity,
        }
    }
}

impl Display for Condition {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Condition::Above => "a",
                Condition::AboveOrEqual => "ae",
                Condition::Below => "b",
                Condition::BelowOrEqual => "be",
                Condition::Equal => "e",
                Condition::NotEqual => "ne",
                Condition::Greater => "g",
                Condition::GreaterOrEqual => "ge",
                Condition::Less => "l",
                Condition::LessOrEqual => "le",
                Condition::Parity => "p",
                Condition::NotParity => "np",
            }
        )
    }
}
