///|
/// Portable RNG, intentionally distinct from Torch RNG.
pub struct Rng {
  priv mut state : UInt32
} derive(Debug)

///|
pub fn Rng::new(seed : UInt32) -> Rng {
  let initial = seed ^ 0x9E3779B9U
  { state: if initial == 0 { 1U } else { initial }, }
}

///|
pub fn Rng::state(self : Rng) -> UInt32 {
  self.state
}

///|
pub fn Rng::snapshot(self : Rng) -> Rng {
  { state: self.state, }
}

///|
pub fn Rng::set_state(self : Rng, state : UInt32) -> Unit {
  guard state != 0 else { abort("xorshift state must be nonzero") }
  self.state = state
}

///|
pub fn Rng::uniform(self : Rng) -> Double {
  let mut x = self.state
  x = x ^ (x << 13)
  x = x ^ (x >> 17)
  x = x ^ (x << 5)
  self.state = x
  (x.to_double() + 0.5) / 4294967296.0
}

///|
pub fn Rng::index(self : Rng, length : Int32) -> Int32 {
  guard length > 0 else { abort("empty sampling pool") }
  (self.uniform() * length.to_double()).to_int()
}

///|
pub fn Rng::normal(self : Rng) -> Double {
  (-2.0 * @math.ln(self.uniform())).sqrt() *
  @math.cos(6.283185307179586 * self.uniform())
}

///|
pub fn[T] Rng::shuffle(self : Rng, values : Array[T]) -> Unit {
  for i = values.length() - 1; i > 0; i = i - 1 {
    let j = self.index(i + 1)
    let saved = values[i]
    values[i] = values[j]
    values[j] = saved
  }
}

///|
/// SiLU hidden layers, row-major [output,input] weights followed by biases.
pub(all) struct MLP {
  dimensions : Array[Int32]
  parameters : Array[Double]
} derive(Debug)

///|
pub fn MLP::new(
  dimensions : Array[Int32],
  rng : Rng,
  zero_last~ : Bool = false,
) -> MLP {
  guard dimensions.length() >= 2 && dimensions.all(n => n > 0) else {
    abort("invalid MLP dimensions")
  }
  let parameters = []
  for layer in 0..<(dimensions.length() - 1) {
    let nin = dimensions[layer]
    let nout = dimensions[layer + 1]
    let bound = 1.0 / nin.to_double().sqrt()
    let zero = zero_last && layer == dimensions.length() - 2
    for _ in 0..<(nin * nout + nout) {
      parameters.push(
        if zero {
          0.0
        } else {
          (2.0 * rng.uniform() - 1) * bound
        },
      )
    }
  }
  { dimensions: dimensions.copy(), parameters, }
}

///|
pub fn MLP::bind(self : MLP, tape : Tape) -> Array[Value] {
  tape.vector(self.parameters)
}

///|
pub fn MLP::forward(
  self : MLP,
  tape : Tape,
  parameters : Array[Value],
  inputs : Array[Value],
) -> Array[Value] {
  guard inputs.length() == self.dimensions[0] &&
    parameters.length() == self.parameters.length() else {
    abort("MLP shape mismatch")
  }
  let mut values = inputs
  let mut offset = 0
  for layer in 0..<(self.dimensions.length() - 1) {
    let nin = self.dimensions[layer]
    let nout = self.dimensions[layer + 1]
    let linear = tape.linear(parameters, values, nout, offset~)
    let next = if layer < self.dimensions.length() - 2 {
      linear.map(value => tape.silu(value))
    } else {
      linear
    }
    offset += nin * nout + nout
    values = next
  }
  values
}

///|
pub fn MLP::predict(self : MLP, inputs : Array[Double]) -> Array[Double] {
  guard inputs.length() == self.dimensions[0] else {
    abort("MLP input shape mismatch")
  }
  let mut values = inputs
  let mut offset = 0
  for layer in 0..<(self.dimensions.length() - 1) {
    let nin = self.dimensions[layer]
    let nout = self.dimensions[layer + 1]
    values = Array::makei(nout, row => {
      let mut value = self.parameters[offset + nin * nout + row]
      for column in 0..<nin {
        value += self.parameters[offset + row * nin + column] * values[column]
      }
      if layer < self.dimensions.length() - 2 {
        let sigmoid = if value >= 0 {
          1.0 / (1.0 + @math.exp(-value))
        } else {
          let e = @math.exp(value)
          e / (1.0 + e)
        }
        value * sigmoid
      } else {
        value
      }
    })
    offset += nin * nout + nout
  }
  values
}

///|
pub fn MLP::snapshot(self : MLP) -> MLP {
  { dimensions: self.dimensions.copy(), parameters: self.parameters.copy(), }
}

///|
pub(all) struct Normalization {
  mean : Array[Double]
  scale : Array[Double]
} derive(Debug)

///|
pub fn finite(value : Double) -> Bool {
  value == value && value - value == 0
}

///|
pub fn fit_normalization(
  rows : Array[Array[Double]],
  floor : Double,
) -> Normalization {
  guard rows.length() > 0 && floor > 0 else {
    abort("normalization requires data and positive floor")
  }
  let width = rows[0].length()
  let mean = Array::make(width, 0.0)
  let variance = Array::make(width, 0.0)
  for row in rows {
    guard row.length() == width else { abort("ragged normalization data") }
    for j in 0..<width {
      mean[j] += row[j] / rows.length().to_double()
    }
  }
  for row in rows {
    for j in 0..<width {
      let delta = row[j] - mean[j]
      variance[j] += delta * delta / rows.length().to_double()
    }
  }
  { mean, scale: variance.map(v => @cmp.maximum(v.sqrt(), floor)), }
}

///|
pub fn Normalization::apply(
  self : Normalization,
  tape : Tape,
  inputs : Array[Value],
) -> Array[Value] {
  guard inputs.length() == self.mean.length() else {
    abort("normalization shape mismatch")
  }
  Array::makei(inputs.length(), i => {
    tape.scale(
      tape.sub(inputs[i], tape.scalar(self.mean[i])),
      1 / self.scale[i],
    )
  })
}
