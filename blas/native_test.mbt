///|
/// Native differential execution is deferred. This fixture fails explicitly until a caller provides an LP64 provider path.
test "native CBLAS agrees with portable deterministic matrices" {
  guard @env.get_env_var("MOONNUM_BLAS_LIBRARY") is Some(path) else {
    fail("deferred BLAS differential requires MOONNUM_BLAS_LIBRARY")
  }
  let native = @blas.Backend::load(path)
  let portable = @blas.Backend::portable()
  assert_true(native.is_native())
  let a = Array::makei(35, i => (i % 11 - 5).to_double() / 7)
  let b = Array::makei(21, i => (i % 7 - 3).to_double() / 9)
  let expected = portable.gemm(a, b, 5, 3, 7)
  let actual = native.gemm(a, b, 5, 3, 7)
  for i in 0..<actual.length() {
    assert_true((actual[i] - expected[i]).abs() < 0.000000000001)
  }
  let af = a.map(v => v.to_float())
  let bf = b.map(v => v.to_float())
  let expected32 = portable.gemm_f32(af, bf, 5, 3, 7)
  let actual32 = native.gemm_f32(af, bf, 5, 3, 7)
  for i in 0..<actual32.length() {
    assert_true((actual32[i] - expected32[i]).abs() < 0.00001F)
  }
}
