; ModuleID = 'main'
source_filename = "main"
target triple = "x86_64-pc-linux-gnu"

define void @main() {
entry:
  %pow = call double @pow(double 2.000000e+00, double 3.000000e+00)
  %pow_i64_res = fptosi double %pow to i64
  %pow_rhs = sitofp i64 %pow_i64_res to double
  %pow1 = call double @pow(double 2.000000e+00, double %pow_rhs)
  %pow_i64_res2 = fptosi double %pow1 to i64
  %i_sum = sub i64 %pow_i64_res2, 64
  call void @exit(i64 %i_sum)
  ret void
}

declare void @exit(i64)

declare double @pow(double, double)
