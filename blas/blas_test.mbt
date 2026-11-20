///|
test "portable BLAS multiplies row-major and transposed inputs in both precisions" {
  let backend = @blas.Backend::portable()
  assert_false(backend.is_native())
  assert_eq(
    backend.gemm(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
      2,
      2,
      3,
    ),
    [4.0, 5.0, 10.0, 11.0],
  )
  assert_eq(
    backend.gemm(
      [1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
      [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
      2,
      2,
      3,
      transpose_a=true,
      transpose_b=true,
    ),
    [4.0, 5.0, 10.0, 11.0],
  )
  assert_eq(
    backend.gemm_f32(
      [16777216.0F, 1.0F, -16777216.0F],
      [1.0F, 1.0F, 1.0F],
      1,
      1,
      3,
    ),
    [0.0F],
  )
  assert_true(
    try backend.gemm([1.0], [2.0], 2, 2, 2) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
