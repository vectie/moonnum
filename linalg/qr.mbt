///|
/// Column-pivoted Householder QR result. Rank-deficient solutions set free pivoted coordinates to zero; they are basic least-squares solutions, not Moore-Penrose minimum-norm solutions.
pub(all) struct LeastSquares {
  solution : Array[Array[Double]]
  rank : Int32
  residual_sum_squares : Array[Double]
  condition_estimate : Double
  pivots : Array[Int32]
} derive(Debug)

///|
/// Stable scaled Euclidean norm avoids overflow from squaring large coordinates and explicitly rejects norms outside finite Double range.
fn column_norm(
  a : Array[Array[Double]],
  first : Int32,
  column : Int32,
) -> Double raise LinearAlgebraError {
  let mut scale = 0.0
  let mut sum = 1.0
  for i in first..<a.length() {
    let x = a[i][column].abs()
    guard finite(x) else { raise Invalid("nonfinite QR column") }
    if x != 0 {
      if scale < x {
        let ratio = scale / x
        sum = 1 + sum * ratio * ratio
        scale = x
      } else {
        let ratio = x / scale
        sum += ratio * ratio
      }
    }
  }
  let norm = if scale == 0 { 0.0 } else { scale * sum.sqrt() }
  guard finite(norm) else { raise Invalid("QR column norm overflow") }
  norm
}

///|
/// Rank-revealing column-pivoted QR supports tall, square and underdetermined finite matrices without forming normal equations. rcond is relative to the largest initial column norm.
pub fn least_squares(
  a : Array[Array[Double]],
  b : Array[Array[Double]],
  rcond~ : Double = 0.000000000001,
) -> LeastSquares raise LinearAlgebraError {
  let (m, n) = matrix_shape(a)
  let (bm, nrhs) = matrix_shape(b)
  guard bm == m && rcond >= 0 && finite(rcond) else {
    raise Invalid("least-squares shape/rcond")
  }
  let qr = a.map(row => row.copy())
  let rhs = b.map(row => row.copy())
  let pivots = Array::makei(n, i => i)
  let limit = if m < n { m } else { n }
  let mut rank = 0
  let mut largest = 0.0
  let mut smallest = 1.0 / 0.0
  for k in 0..<limit {
    let mut pivot = k
    let mut norm = column_norm(qr, k, k)
    for j in (k + 1)..<n {
      let candidate = column_norm(qr, k, j)
      if candidate > norm {
        pivot = j
        norm = candidate
      }
    }
    if k == 0 {
      largest = norm
    }
    if norm == 0 || norm <= largest * rcond {
      break
    }
    if pivot != k {
      for i in 0..<m {
        let v = qr[i][k]
        qr[i][k] = qr[i][pivot]
        qr[i][pivot] = v
      }
      let p = pivots[k]
      pivots[k] = pivots[pivot]
      pivots[pivot] = p
    }
    let alpha = if qr[k][k] >= 0 { -norm } else { norm }
    // Normalize the Householder vector before its dot products.
    let v = Array::makei(m - k, i => qr[k + i][k] / norm)
    v[0] -= alpha / norm
    let mut squared = 0.0
    for value in v {
      squared += value * value
    }
    let beta = 2.0 / squared
    for j in k..<n {
      let mut dot = 0.0
      for i in 0..<v.length() {
        dot += v[i] * qr[k + i][j]
      }
      guard finite(dot) else { raise Invalid("QR reflection overflow") }
      for i in 0..<v.length() {
        qr[k + i][j] -= beta * v[i] * dot
        guard finite(qr[k + i][j]) else {
          raise Invalid("QR reflection overflow")
        }
      }
    }
    for j in 0..<nrhs {
      let mut dot = 0.0
      for i in 0..<v.length() {
        dot += v[i] * rhs[k + i][j]
      }
      guard finite(dot) else {
        raise Invalid("QR right-hand-side reflection overflow")
      }
      for i in 0..<v.length() {
        rhs[k + i][j] -= beta * v[i] * dot
        guard finite(rhs[k + i][j]) else {
          raise Invalid("QR right-hand-side reflection overflow")
        }
      }
    }
    qr[k][k] = alpha
    for i in (k + 1)..<m {
      qr[i][k] = 0
    }
    if alpha.abs() < smallest {
      smallest = alpha.abs()
    }
    rank += 1
  }
  let pivot_solution = Array::makei(n, _ => Array::make(nrhs, 0.0))
  for i = rank - 1; i >= 0; i = i - 1 {
    for j in 0..<nrhs {
      let mut value = rhs[i][j]
      for k in (i + 1)..<rank {
        value -= qr[i][k] * pivot_solution[k][j]
      }
      pivot_solution[i][j] = value / qr[i][i]
    }
  }
  let solution = Array::makei(n, _ => Array::make(nrhs, 0.0))
  for i in 0..<n {
    solution[pivots[i]] = pivot_solution[i]
  }
  let residual_sum_squares = Array::make(nrhs, 0.0)
  for i in 0..<m {
    for j in 0..<nrhs {
      let mut predicted = 0.0
      for k in 0..<n {
        predicted += a[i][k] * solution[k][j]
      }
      let delta = predicted - b[i][j]
      residual_sum_squares[j] += delta * delta
    }
  }
  guard solution.all(row => row.all(finite)) && residual_sum_squares.all(finite) else {
    raise Invalid("least-squares overflow")
  }
  {
    solution,
    rank,
    residual_sum_squares,
    condition_estimate: if rank == 0 {
      1.0 / 0.0
    } else {
      largest / smallest
    },
    pivots,
  }
}
