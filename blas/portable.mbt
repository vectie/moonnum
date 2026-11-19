///|
/// Non-native targets expose the same selection API and reject dynamic loading explicitly.
enum Handle {
  UnavailableHandle
}

///|
fn load_handle(_path : String) -> Handle raise BlasError {
  raise Unavailable(
    "dynamic CBLAS loading requires native target; select Backend::portable()",
  )
}

///|
fn gemm64(
  _handle : Handle,
  _a : FixedArray[Double],
  _b : FixedArray[Double],
  _c : FixedArray[Double],
  _m : Int32,
  _n : Int32,
  _k : Int32,
  _ta : Bool,
  _tb : Bool,
) -> Unit {
  abort("unreachable non-native CBLAS")
}

///|
fn gemm32(
  _handle : Handle,
  _a : FixedArray[Float],
  _b : FixedArray[Float],
  _c : FixedArray[Float],
  _m : Int32,
  _n : Int32,
  _k : Int32,
  _ta : Bool,
  _tb : Bool,
) -> Unit {
  abort("unreachable non-native CBLAS")
}
