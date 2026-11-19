///|
/// GC-managed dynamic-library ownership; only the shared library is finalized.
type Handle

///|
#borrow(path)
extern "c" fn load_native(path : Bytes) -> Handle = "moonnum_blas_load"

///|
#borrow(handle)
extern "c" fn error_native(handle : Handle) -> Bytes = "moonnum_blas_error"

///|
#borrow(handle, a, b, c)
extern "c" fn gemm64(
  handle : Handle,
  a : FixedArray[Double],
  b : FixedArray[Double],
  c : FixedArray[Double],
  m : Int32,
  n : Int32,
  k : Int32,
  ta : Bool,
  tb : Bool,
) -> Unit = "moonnum_blas_dgemm"

///|
#borrow(handle, a, b, c)
extern "c" fn gemm32(
  handle : Handle,
  a : FixedArray[Float],
  b : FixedArray[Float],
  c : FixedArray[Float],
  m : Int32,
  n : Int32,
  k : Int32,
  ta : Bool,
  tb : Bool,
) -> Unit = "moonnum_blas_sgemm"

///|
fn load_handle(path : String) -> Handle raise BlasError {
  let handle = load_native(@utf8.encode(path))
  let error = @utf8.decode_lossy(error_native(handle))
  guard error == "" else { raise Unavailable(error) }
  handle
}
