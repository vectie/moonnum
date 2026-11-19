///|
/// An explicitly selected backend. portable() never probes host libraries; load(path) either loads both CBLAS precisions or raises.
pub struct Backend {
  priv handle : Handle?
  priv path : String?
}

///|
pub suberror BlasError {
  Invalid(String)
  Unavailable(String)
} derive(Debug)

///|
pub fn Backend::portable() -> Backend {
  { handle: None, path: None, }
}

///|
pub fn Backend::load(path : String) -> Backend raise BlasError {
  guard path != "" && !path.contains("\u0000") else {
    raise Invalid("BLAS path is empty or contains NUL")
  }
  let handle = load_handle(path)
  { handle: Some(handle), path: Some(path), }
}

///|
pub fn Backend::is_native(self : Backend) -> Bool {
  self.handle is Some(_)
}

///|
pub fn Backend::library_path(self : Backend) -> String? {
  self.path
}

///|
fn checked_shape(
  m : Int32,
  n : Int32,
  k : Int32,
  alen : Int32,
  blen : Int32,
) -> Unit raise BlasError {
  guard m >= 0 &&
    n >= 0 &&
    k >= 0 &&
    (m == 0 || k <= 2147483647 / m) &&
    (k == 0 || n <= 2147483647 / k) &&
    (m == 0 || n <= 2147483647 / m) &&
    alen == m * k &&
    blen == k * n else {
    raise Invalid("GEMM dimensions/lengths overflow or mismatch")
  }
}

///|
/// Row-major matrix multiplication. Inputs are call-local copies when passed to native BLAS. Native reduction/FMA order belongs to the loaded provider and can differ from portable accumulation.
pub fn Backend::gemm(
  self : Backend,
  a : Array[Double],
  b : Array[Double],
  m : Int32,
  n : Int32,
  k : Int32,
  transpose_a~ : Bool = false,
  transpose_b~ : Bool = false,
) -> Array[Double] raise BlasError {
  checked_shape(m, n, k, a.length(), b.length())
  let result = FixedArray::make(m * n, 0.0)
  if m == 0 || n == 0 || k == 0 {
    return Array::makei(m * n, i => result[i])
  }
  match self.handle {
    Some(handle) =>
      gemm64(
        handle,
        FixedArray::makei(a.length(), i => a[i]),
        FixedArray::makei(b.length(), i => b[i]),
        result,
        m,
        n,
        k,
        transpose_a,
        transpose_b,
      )
    None =>
      for row in 0..<m {
        for column in 0..<n {
          let mut sum = 0.0
          for inner in 0..<k {
            sum += a[if transpose_a {
                inner * m + row
              } else {
                row * k + inner
              }] *
              b[if transpose_b {
                column * k + inner
              } else {
                inner * n + column
              }]
          }
          result[row * n + column] = sum
        }
      }
  }
  Array::makei(m * n, i => result[i])
}

///|
pub fn Backend::gemm_f32(
  self : Backend,
  a : Array[Float],
  b : Array[Float],
  m : Int32,
  n : Int32,
  k : Int32,
  transpose_a~ : Bool = false,
  transpose_b~ : Bool = false,
) -> Array[Float] raise BlasError {
  checked_shape(m, n, k, a.length(), b.length())
  let result = FixedArray::make(m * n, 0.0F)
  if m == 0 || n == 0 || k == 0 {
    return Array::makei(m * n, i => result[i])
  }
  match self.handle {
    Some(handle) =>
      gemm32(
        handle,
        FixedArray::makei(a.length(), i => a[i]),
        FixedArray::makei(b.length(), i => b[i]),
        result,
        m,
        n,
        k,
        transpose_a,
        transpose_b,
      )
    None =>
      for row in 0..<m {
        for column in 0..<n {
          let mut sum = 0.0F
          for inner in 0..<k {
            let product = a[if transpose_a {
                inner * m + row
              } else {
                row * k + inner
              }] *
              b[if transpose_b {
                column * k + inner
              } else {
                inner * n + column
              }]
            sum = sum + product
          }
          result[row * n + column] = sum
        }
      }
  }
  Array::makei(m * n, i => result[i])
}
