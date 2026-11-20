///|
fn broadcast_shape(
  a : Array[Int32],
  b : Array[Int32],
) -> Array[Int32] raise TensorError {
  let rank = if a.length() > b.length() { a.length() } else { b.length() }
  Array::makei(rank, i => {
    let ai = i - (rank - a.length())
    let bi = i - (rank - b.length())
    let x = if ai < 0 { 1 } else { a[ai] }
    let y = if bi < 0 { 1 } else { b[bi] }
    guard x == y || x == 1 || y == 1 else {
      raise Invalid("incompatible broadcast")
    }
    if x == 1 {
      y
    } else {
      x
    }
  })
}

///|
pub fn Tensor::map(self : Tensor, operation : (Double) -> Double) -> Tensor {
  try! Tensor::from_array(
    self.to_array().map(operation),
    self.dimensions,
    dtype=self.dtype(),
  )
}

///|
pub fn Tensor::binary(
  self : Tensor,
  other : Tensor,
  operation : (Double, Double) -> Double,
) -> Tensor raise TensorError {
  guard self.dtype() == other.dtype() else {
    raise Invalid("mixed dtypes require explicit cast")
  }
  let shape = broadcast_shape(self.dimensions, other.dimensions)
  let a = self.broadcast_to(shape)
  let b = other.broadcast_to(shape)
  Tensor::from_array(
    Array::makei(element_count(shape), i => {
      operation(a.read_at(a.flat_address(i)), b.read_at(b.flat_address(i)))
    }),
    shape,
    dtype=self.dtype(),
  )
}

///|
pub fn Tensor::cast(self : Tensor, dtype : DType) -> Tensor {
  try! Tensor::from_array(self.to_array(), self.dimensions, dtype~)
}

///|
pub fn Tensor::add(self : Tensor, other : Tensor) -> Tensor raise TensorError {
  self.binary(other, (a, b) => a + b)
}

///|
pub fn Tensor::sub(self : Tensor, other : Tensor) -> Tensor raise TensorError {
  self.binary(other, (a, b) => a - b)
}

///|
pub fn Tensor::mul(self : Tensor, other : Tensor) -> Tensor raise TensorError {
  self.binary(other, (a, b) => a * b)
}

///|
pub fn Tensor::div(self : Tensor, other : Tensor) -> Tensor raise TensorError {
  self.binary(other, (a, b) => a / b)
}

///|
pub fn Tensor::scale(self : Tensor, factor : Double) -> Tensor {
  self.map(value => value * factor)
}

///|
pub fn Tensor::exp(self : Tensor) -> Tensor {
  self.map(value => @math.exp(value))
}

///|
pub fn Tensor::log(self : Tensor) -> Tensor {
  self.map(value => @math.ln(value))
}

///|
pub fn Tensor::tanh(self : Tensor) -> Tensor {
  self.map(value => @math.tanh(value))
}

///|
pub fn Tensor::silu(self : Tensor) -> Tensor {
  self.map(value => {
    let sigmoid = if value >= 0 {
      1.0 / (1 + @math.exp(-value))
    } else {
      let e = @math.exp(value)
      e / (1 + e)
    }
    value * sigmoid
  })
}

///|
fn rounded(dtype : DType, value : Double) -> Double {
  match dtype {
    F32 => Float::from_double(value).to_double()
    F64 => value
  }
}

///|
/// Left-to-right reduction rounds each addition in the declared dtype; no hidden higher-precision F32 accumulator.
pub fn Tensor::sum(
  self : Tensor,
  axis? : Int32,
  keepdim~ : Bool = false,
) -> Tensor raise TensorError {
  match axis {
    None => {
      let mut total = 0.0
      for value in self.to_array() {
        total = rounded(self.dtype(), total + value)
      }
      let shape = if keepdim { Array::make(self.rank(), 1) } else { [] }
      Tensor::from_array([total], shape, dtype=self.dtype())
    }
    Some(axis) => {
      let axis = axis_index(axis, self.rank())
      let shape = self.dimensions.copy()
      if keepdim {
        shape[axis] = 1
      } else {
        ignore(shape.remove(axis))
      }
      let result = Tensor::zeros(shape, dtype=self.dtype())
      let coordinates = Array::make(self.rank(), 0)
      for flat in 0..<self.numel() {
        let mut remaining = flat
        for i = self.rank() - 1; i >= 0; i = i - 1 {
          coordinates[i] = remaining % self.dimensions[i]
          remaining /= self.dimensions[i]
        }
        let target = coordinates.copy()
        if keepdim {
          target[axis] = 0
        } else {
          ignore(target.remove(axis))
        }
        let address = result.address(target)
        result.write_at(
          address,
          result.read_at(address) + self.read_at(self.flat_address(flat)),
        )
      }
      result
    }
  }
}

///|
pub fn Tensor::mean(
  self : Tensor,
  axis? : Int32,
  keepdim~ : Bool = false,
) -> Tensor raise TensorError {
  let count = match axis {
    None => self.numel()
    Some(a) => self.dimensions[axis_index(a, self.rank())]
  }
  self.sum(axis?, keepdim~).scale(1.0 / count.to_double())
}

///|
/// NumPy-style matmul shape rules, including vector endpoints and broadcast batch dimensions; no implicit dtype promotion.
pub fn Tensor::matmul(
  self : Tensor,
  other : Tensor,
  backend? : @blas.Backend,
) -> Tensor raise {
  guard self.rank() > 0 && other.rank() > 0 && self.dtype() == other.dtype() else {
    raise Invalid("matmul rank/dtype")
  }
  let left_vector = self.rank() == 1
  let right_vector = other.rank() == 1
  let a = if left_vector { self.reshape([1, self.dimensions[0]]) } else { self }
  let b = if right_vector {
    other.reshape([other.dimensions[0], 1])
  } else {
    other
  }
  let ar = a.rank()
  let br = b.rank()
  let m = a.dimensions[ar - 2]
  let k = a.dimensions[ar - 1]
  let n = b.dimensions[br - 1]
  guard k == b.dimensions[br - 2] else {
    raise Invalid("matmul inner dimensions")
  }
  let batch = broadcast_shape(
    a.dimensions[:ar - 2].to_owned(),
    b.dimensions[:br - 2].to_owned(),
  )
  let ashape = batch.copy()
  ashape.append([m, k])
  let bshape = batch.copy()
  bshape.append([k, n])
  let av = a.broadcast_to(ashape)
  let bv = b.broadcast_to(bshape)
  let result_shape = batch.copy()
  result_shape.append([m, n])
  let result = Tensor::zeros(result_shape, dtype=self.dtype())
  for item in 0..<element_count(batch) {
    let coordinates = Array::make(batch.length(), 0)
    let mut remaining = item
    for i = batch.length() - 1; i >= 0; i = i - 1 {
      coordinates[i] = remaining % batch[i]
      remaining /= batch[i]
    }
    let mut abase = av.offset
    let mut bbase = bv.offset
    for i in 0..<batch.length() {
      abase += coordinates[i] * av.steps[i]
      bbase += coordinates[i] * bv.steps[i]
    }
    if backend is Some(selected) {
      let packed_a = Array::makei(m * k, i => {
        av.read_at(
          abase +
          i / k * av.steps[batch.length()] +
          i % k * av.steps[batch.length() + 1],
        )
      })
      let packed_b = Array::makei(k * n, i => {
        bv.read_at(
          bbase +
          i / n * bv.steps[batch.length()] +
          i % n * bv.steps[batch.length() + 1],
        )
      })
      let values = match self.dtype() {
        F64 => selected.gemm(packed_a, packed_b, m, n, k)
        F32 =>
          selected
          .gemm_f32(
            packed_a.map(x => Float::from_double(x)),
            packed_b.map(x => Float::from_double(x)),
            m,
            n,
            k,
          )
          .map(x => x.to_double())
      }
      for i, value in values {
        result.write_at(item * m * n + i, value)
      }
      continue
    }
    for row in 0..<m {
      for column in 0..<n {
        let mut sum = 0.0
        for inner in 0..<k {
          let product = rounded(
            self.dtype(),
            av.read_at(
              abase +
              row * av.steps[batch.length()] +
              inner * av.steps[batch.length() + 1],
            ) *
            bv.read_at(
              bbase +
              inner * bv.steps[batch.length()] +
              column * bv.steps[batch.length() + 1],
            ),
          )
          sum = rounded(self.dtype(), sum + product)
        }
        result.write_at((item * m + row) * n + column, sum)
      }
    }
  }
  let shape = result_shape.copy()
  if right_vector {
    ignore(shape.remove(shape.length() - 1))
  }
  if left_vector {
    ignore(shape.remove(batch.length()))
  }
  result.view(shape)
}
