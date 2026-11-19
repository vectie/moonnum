///|
/// Dense checked linear algebra. Inputs are copied; callers retain ownership.
pub suberror LinearAlgebraError {
  Invalid(String)
  Singular(Int32)
} derive(Debug)

///|
fn finite(x : Double) -> Bool {
  x == x && x - x == 0
}

///|
fn matrix_shape(
  a : Array[Array[Double]],
) -> (Int32, Int32) raise LinearAlgebraError {
  guard !a.is_empty() && !a[0].is_empty() else { raise Invalid("empty matrix") }
  let n = a[0].length()
  guard a.all(row => row.length() == n && row.all(finite)) else {
    raise Invalid("ragged or nonfinite matrix")
  }
  (a.length(), n)
}

///|
/// Scaled partial-pivot LU with multiple right-hand sides; rejects numerically singular pivots relative to row scale.
pub fn solve(
  a : Array[Array[Double]],
  b : Array[Array[Double]],
  tolerance~ : Double = 0.000000000001,
) -> Array[Array[Double]] raise LinearAlgebraError {
  let (m, n) = matrix_shape(a)
  let (bm, nrhs) = matrix_shape(b)
  guard m == n && bm == n && tolerance >= 0 && finite(tolerance) else {
    raise Invalid("solve shape/tolerance")
  }
  let lu = a.map(row => row.copy())
  let x = b.map(row => row.copy())
  let scales = lu.map(row => {
    let mut scale = 0.0
    for v in row {
      if v.abs() > scale {
        scale = v.abs()
      }
    }
    scale
  })
  for k in 0..<n {
    let mut pivot = k
    let mut ratio = -1.0
    for i in k..<n {
      let r = if scales[i] == 0 { 0.0 } else { lu[i][k].abs() / scales[i] }
      if r > ratio {
        ratio = r
        pivot = i
      }
    }
    guard scales[pivot] > 0 && lu[pivot][k].abs() > tolerance * scales[pivot] else {
      raise Singular(k)
    }
    if pivot != k {
      let row = lu[k]
      lu[k] = lu[pivot]
      lu[pivot] = row
      let rhs = x[k]
      x[k] = x[pivot]
      x[pivot] = rhs
      let scale = scales[k]
      scales[k] = scales[pivot]
      scales[pivot] = scale
    }
    for i in (k + 1)..<n {
      let factor = lu[i][k] / lu[k][k]
      lu[i][k] = factor
      for j in (k + 1)..<n {
        lu[i][j] -= factor * lu[k][j]
      }
      for j in 0..<nrhs {
        x[i][j] -= factor * x[k][j]
      }
    }
  }
  for i = n - 1; i >= 0; i = i - 1 {
    for j in 0..<nrhs {
      for k in (i + 1)..<n {
        x[i][j] -= lu[i][k] * x[k][j]
      }
      x[i][j] /= lu[i][i]
    }
  }
  guard x.all(row => row.all(finite)) else { raise Invalid("solve overflow") }
  x
}
