///|
/// Explicit storage precision. F32 stores actual MoonBit Float values; accessors widen to Double without changing storage precision.
pub(all) enum DType {
  F32
  F64
} derive(Debug, Eq)

///|
enum Storage {
  Single(Array[Float])
  DoublePrecision(Array[Double])
} derive(Debug)

///|
/// Views share storage, including negative and zero strides. Metadata accessors return copies so callers cannot invalidate the view.
pub struct Tensor {
  priv storage : Storage
  priv dimensions : Array[Int32]
  priv steps : Array[Int32]
  priv offset : Int32
} derive(Debug)

///|
pub suberror TensorError {
  Invalid(String)
} derive(Debug)

///|
fn element_count(shape : Array[Int32]) -> Int32 raise TensorError {
  let mut count = 1
  for size in shape {
    guard size >= 0 && (size == 0 || count <= 2147483647 / size) else {
      raise Invalid("negative or overflowing shape")
    }
    count *= size
  }
  count
}

///|
fn row_strides(shape : Array[Int32]) -> Array[Int32] {
  let strides = Array::make(shape.length(), 1)
  let mut stride = 1L
  for i = shape.length() - 1; i >= 0; i = i - 1 {
    // Only empty tensors can have a suffix larger than the addressable storage.
    strides[i] = if stride > 2147483647L { 0 } else { stride.to_int() }
    if stride > 2147483647L {
      stride = if shape[i] == 0 { 0L } else { 2147483648L }
    } else {
      stride *= shape[i].to_int64()
    }
  }
  strides
}

///|
pub fn Tensor::from_array(
  values : Array[Double],
  shape : Array[Int32],
  dtype~ : DType = F64,
) -> Tensor raise TensorError {
  guard values.length() == element_count(shape) else {
    raise Invalid("data length differs from shape")
  }
  let storage = match dtype {
    F32 => Single(values.map(x => x.to_float()))
    F64 => DoublePrecision(values.copy())
  }
  { storage, dimensions: shape.copy(), steps: row_strides(shape), offset: 0, }
}

///|
pub fn Tensor::from_f32(
  values : Array[Float],
  shape : Array[Int32],
) -> Tensor raise TensorError {
  guard values.length() == element_count(shape) else {
    raise Invalid("data length differs from shape")
  }
  {
    storage: Single(values.copy()),
    dimensions: shape.copy(),
    steps: row_strides(shape),
    offset: 0,
  }
}

///|
pub fn Tensor::full(
  shape : Array[Int32],
  value : Double,
  dtype~ : DType = F64,
) -> Tensor raise TensorError {
  Tensor::from_array(Array::make(element_count(shape), value), shape, dtype~)
}

///|
pub fn Tensor::zeros(
  shape : Array[Int32],
  dtype~ : DType = F64,
) -> Tensor raise TensorError {
  Tensor::full(shape, 0, dtype~)
}

///|
pub fn Tensor::scalar(value : Double, dtype~ : DType = F64) -> Tensor {
  try! Tensor::from_array([value], [], dtype~)
}

///|
pub fn Tensor::dtype(self : Tensor) -> DType {
  match self.storage {
    Single(_) => F32
    DoublePrecision(_) => F64
  }
}

///|
pub fn Tensor::shape(self : Tensor) -> Array[Int32] {
  self.dimensions.copy()
}

///|
pub fn Tensor::strides(self : Tensor) -> Array[Int32] {
  self.steps.copy()
}

///|
pub fn Tensor::rank(self : Tensor) -> Int32 {
  self.dimensions.length()
}

///|
pub fn Tensor::numel(self : Tensor) -> Int32 {
  try! element_count(self.dimensions)
}

///|
pub fn Tensor::storage_offset(self : Tensor) -> Int32 {
  self.offset
}

///|
fn Tensor::read_at(self : Tensor, index : Int32) -> Double {
  match self.storage {
    Single(values) => values[index].to_double()
    DoublePrecision(values) => values[index]
  }
}

///|
fn Tensor::write_at(self : Tensor, index : Int32, value : Double) -> Unit {
  match self.storage {
    Single(values) => values[index] = value.to_float()
    DoublePrecision(values) => values[index] = value
  }
}

///|
fn Tensor::address(
  self : Tensor,
  coordinates : Array[Int32],
) -> Int32 raise TensorError {
  guard coordinates.length() == self.rank() else {
    raise Invalid("coordinate rank")
  }
  let mut offset = self.offset
  for i in 0..<coordinates.length() {
    let index = if coordinates[i] < 0 {
      coordinates[i] + self.dimensions[i]
    } else {
      coordinates[i]
    }
    guard index >= 0 && index < self.dimensions[i] else {
      raise Invalid("coordinate out of bounds")
    }
    offset += index * self.steps[i]
  }
  offset
}

///|
fn Tensor::flat_address(self : Tensor, flat : Int32) -> Int32 {
  let mut index = flat
  let mut offset = self.offset
  for axis = self.rank() - 1; axis >= 0; axis = axis - 1 {
    let size = self.dimensions[axis]
    offset += index % size * self.steps[axis]
    index /= size
  }
  offset
}

///|
pub fn Tensor::get(
  self : Tensor,
  coordinates : Array[Int32],
) -> Double raise TensorError {
  self.read_at(self.address(coordinates))
}

///|
/// Setting a zero-stride broadcast view intentionally changes its shared underlying element.
pub fn Tensor::set(
  self : Tensor,
  coordinates : Array[Int32],
  value : Double,
) -> Unit raise TensorError {
  self.write_at(self.address(coordinates), value)
}

///|
pub fn Tensor::to_array(self : Tensor) -> Array[Double] {
  Array::makei(self.numel(), i => self.read_at(self.flat_address(i)))
}

///|
pub fn Tensor::is_contiguous(self : Tensor) -> Bool {
  if self.numel() == 0 {
    return true
  }
  let mut stride = 1
  for axis = self.rank() - 1; axis >= 0; axis = axis - 1 {
    if self.dimensions[axis] > 1 && self.steps[axis] != stride {
      return false
    }
    stride *= self.dimensions[axis]
  }
  true
}

///|
pub fn Tensor::clone(self : Tensor) -> Tensor {
  try! Tensor::from_array(self.to_array(), self.dimensions, dtype=self.dtype())
}

///|
pub fn Tensor::contiguous(self : Tensor) -> Tensor {
  if self.is_contiguous() {
    self
  } else {
    self.clone()
  }
}

///|
/// A strict reshape view requires contiguous logical order; reshape materializes other layouts.
pub fn Tensor::view(
  self : Tensor,
  shape : Array[Int32],
) -> Tensor raise TensorError {
  guard self.is_contiguous() && element_count(shape) == self.numel() else {
    raise Invalid("view needs contiguous matching shape")
  }
  {
    storage: self.storage,
    dimensions: shape.copy(),
    steps: row_strides(shape),
    offset: self.offset,
  }
}

///|
pub fn Tensor::reshape(
  self : Tensor,
  shape : Array[Int32],
) -> Tensor raise TensorError {
  self.contiguous().view(shape)
}

///|
fn axis_index(axis : Int32, rank : Int32) -> Int32 raise TensorError {
  let result = if axis < 0 { axis + rank } else { axis }
  guard result >= 0 && result < rank else {
    raise Invalid("axis out of bounds")
  }
  result
}

///|
pub fn Tensor::permute(
  self : Tensor,
  axes : Array[Int32],
) -> Tensor raise TensorError {
  guard axes.length() == self.rank() else { raise Invalid("permutation rank") }
  let seen = Array::make(self.rank(), false)
  let actual = axes.map(axis => {
    let index = axis_index(axis, self.rank())
    guard !seen[index] else { raise Invalid("duplicate permutation axis") }
    seen[index] = true
    index
  })
  {
    storage: self.storage,
    dimensions: actual.map(axis => self.dimensions[axis]),
    steps: actual.map(axis => self.steps[axis]),
    offset: self.offset,
  }
}

///|
pub fn Tensor::transpose(
  self : Tensor,
  first : Int32,
  second : Int32,
) -> Tensor raise TensorError {
  let a = axis_index(first, self.rank())
  let b = axis_index(second, self.rank())
  let axes = Array::makei(self.rank(), i => i)
  axes[a] = b
  axes[b] = a
  self.permute(axes)
}

///|
/// Explicit length avoids ambiguous negative-stop conventions. Negative starts count from the end; negative steps create reversed shared views.
pub fn Tensor::slice(
  self : Tensor,
  axis : Int32,
  start : Int32,
  length : Int32,
  step~ : Int32 = 1,
) -> Tensor raise TensorError {
  let axis = axis_index(axis, self.rank())
  let size = self.dimensions[axis]
  let start = if start < 0 { start + size } else { start }
  guard step != 0 && length >= 0 else { raise Invalid("slice length/step") }
  let last = start.to_int64() + (length.to_int64() - 1) * step.to_int64()
  guard (if length == 0 {
    start >= 0 && start <= size
  } else {
    start >= 0 && start < size && last >= 0L && last < size.to_int64()
  }) else {
    raise Invalid("slice out of bounds")
  }
  let dimensions = self.dimensions.copy()
  let steps = self.steps.copy()
  dimensions[axis] = length
  let sliced_stride = steps[axis].to_int64() * step.to_int64()
  guard sliced_stride >= -2147483648L && sliced_stride <= 2147483647L else {
    raise Invalid("slice stride overflow")
  }
  steps[axis] = sliced_stride.to_int()
  {
    storage: self.storage,
    dimensions,
    steps,
    offset: if length == 0 {
      self.offset
    } else {
      self.offset + start * self.steps[axis]
    },
  }
}

///|
pub fn Tensor::broadcast_to(
  self : Tensor,
  shape : Array[Int32],
) -> Tensor raise TensorError {
  ignore(element_count(shape))
  guard shape.length() >= self.rank() else { raise Invalid("broadcast rank") }
  let steps = Array::make(shape.length(), 0)
  let shift = shape.length() - self.rank()
  for i in 0..<self.rank() {
    let target = i + shift
    guard self.dimensions[i] == shape[target] || self.dimensions[i] == 1 else {
      raise Invalid("broadcast dimension")
    }
    steps[target] = if self.dimensions[i] == 1 { 0 } else { self.steps[i] }
  }
  {
    storage: self.storage,
    dimensions: shape.copy(),
    steps,
    offset: self.offset,
  }
}
