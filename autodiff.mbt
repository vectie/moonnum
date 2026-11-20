///|
/// Reverse-mode scalar node. A tape owns nodes; never mix values from tapes.
pub struct Value {
  id : Int32
  value : Double
} derive(Debug)

///|
struct Node {
  left : Int32
  right : Int32
  dl : Double
  dr : Double
  mut gradient : Double
  block : Int32
}

///|
/// One reverse kernel for a complete dense layer. Outputs are contiguous tape leaves.
struct LinearBlock {
  inputs : Array[Value]
  parameters : Array[Value]
  offset : Int32
  first_output : Int32
  outputs : Int32
}

///|
/// A fresh tape per minibatch releases the entire unrolled computation graph.
pub struct Tape {
  priv nodes : Array[Node]
  priv blocks : Array[LinearBlock]
}

///|
pub fn Tape::new() -> Tape {
  { nodes: [], blocks: [], }
}

///|
pub fn Tape::scalar(self : Tape, value : Double) -> Value {
  self.node(value, -1, -1, 0, 0)
}

///|
fn Tape::node(
  self : Tape,
  value : Double,
  left : Int32,
  right : Int32,
  dl : Double,
  dr : Double,
) -> Value {
  let id = self.nodes.length()
  self.nodes.push({ left, right, dl, dr, gradient: 0, block: -1, })
  { id, value, }
}

///|
pub fn Tape::add(self : Tape, a : Value, b : Value) -> Value {
  self.node(a.value + b.value, a.id, b.id, 1, 1)
}

///|
pub fn Tape::sub(self : Tape, a : Value, b : Value) -> Value {
  self.node(a.value - b.value, a.id, b.id, 1, -1)
}

///|
pub fn Tape::mul(self : Tape, a : Value, b : Value) -> Value {
  self.node(a.value * b.value, a.id, b.id, b.value, a.value)
}

///|
pub fn Tape::div(self : Tape, a : Value, b : Value) -> Value {
  self.node(
    a.value / b.value,
    a.id,
    b.id,
    1 / b.value,
    -a.value / (b.value * b.value),
  )
}

///|
pub fn Tape::scale(self : Tape, a : Value, factor : Double) -> Value {
  self.node(a.value * factor, a.id, -1, factor, 0)
}

///|
pub fn Tape::exp(self : Tape, a : Value) -> Value {
  let value = @math.exp(a.value)
  self.node(value, a.id, -1, value, 0)
}

///|
pub fn Tape::log(self : Tape, a : Value) -> Value {
  self.node(@math.ln(a.value), a.id, -1, 1 / a.value, 0)
}

///|
pub fn Tape::sqrt(self : Tape, a : Value) -> Value {
  let value = a.value.sqrt()
  self.node(value, a.id, -1, 0.5 / value, 0)
}

///|
pub fn Tape::tanh(self : Tape, a : Value) -> Value {
  let value = @math.tanh(a.value)
  self.node(value, a.id, -1, 1 - value * value, 0)
}

///|
pub fn Tape::sigmoid(self : Tape, a : Value) -> Value {
  let value = if a.value >= 0 {
    1.0 / (1.0 + @math.exp(-a.value))
  } else {
    let e = @math.exp(a.value)
    e / (1.0 + e)
  }
  self.node(value, a.id, -1, value * (1 - value), 0)
}

///|
pub fn Tape::silu(self : Tape, a : Value) -> Value {
  self.mul(a, self.sigmoid(a))
}

///|
/// Piecewise clamp, including PyTorch's derivative at the bounds.
pub fn Tape::clamp(
  self : Tape,
  a : Value,
  lower : Double,
  upper : Double,
) -> Value {
  if a.value < lower {
    self.scalar(lower)
  } else if a.value > upper {
    self.scalar(upper)
  } else {
    a
  }
}

///|
pub fn Tape::minimum(self : Tape, a : Value, b : Value) -> Value {
  if a.value < b.value {
    a
  } else if a.value > b.value {
    b
  } else {
    self.scale(self.add(a, b), 0.5)
  }
}

///|
pub fn Tape::maximum(self : Tape, a : Value, b : Value) -> Value {
  if a.value > b.value {
    a
  } else if a.value < b.value {
    b
  } else {
    self.scale(self.add(a, b), 0.5)
  }
}

///|
pub fn Tape::backward(self : Tape, output : Value) -> Unit {
  for node in self.nodes {
    node.gradient = 0
  }
  self.nodes[output.id].gradient = 1
  for i = self.nodes.length() - 1; i >= 0; i = i - 1 {
    let node = self.nodes[i]
    if node.block >= 0 {
      let block = self.blocks[node.block]
      if i == block.first_output {
        let nin = block.inputs.length()
        for row = block.outputs - 1; row >= 0; row = row - 1 {
          let gradient = self.nodes[block.first_output + row].gradient
          if gradient == 0 {
            continue
          }
          for column = nin - 1; column >= 0; column = column - 1 {
            let weight = block.parameters[block.offset + row * nin + column]
            let input = block.inputs[column]
            self.nodes[weight.id].gradient += gradient * input.value
            self.nodes[input.id].gradient += gradient * weight.value
          }
          let bias = block.parameters[block.offset + nin * block.outputs + row]
          self.nodes[bias.id].gradient += gradient
        }
      }
      continue
    }
    if node.gradient == 0 {
      continue
    }
    if node.left >= 0 {
      self.nodes[node.left].gradient += node.gradient * node.dl
    }
    if node.right >= 0 {
      self.nodes[node.right].gradient += node.gradient * node.dr
    }
  }
}

///|
/// Dense kernel stores one output node per row and one reverse block per layer, rather than two scalar nodes per MAC. Parameters retain weights followed by biases.
pub fn Tape::linear(
  self : Tape,
  parameters : Array[Value],
  inputs : Array[Value],
  outputs : Int32,
  offset~ : Int32 = 0,
) -> Array[Value] {
  let nin = inputs.length()
  guard outputs > 0 &&
    nin > 0 &&
    offset >= 0 &&
    offset <= parameters.length() &&
    nin <= 2147483647 / outputs &&
    outputs <= parameters.length() - offset &&
    nin * outputs <= parameters.length() - offset - outputs else {
    abort("linear shape mismatch")
  }
  let block_id = self.blocks.length()
  let first_output = self.nodes.length()
  let values = Array::makei(outputs, row => {
    let mut value = parameters[offset + nin * outputs + row].value
    for column in 0..<nin {
      value += parameters[offset + row * nin + column].value *
        inputs[column].value
    }
    let id = self.nodes.length()
    self.nodes.push({
      left: -1,
      right: -1,
      dl: 0,
      dr: 0,
      gradient: 0,
      block: block_id,
    })
    ({ id, value, } : Value)
  })
  self.blocks.push({
    inputs: inputs.copy(),
    parameters: parameters[offset:offset + nin * outputs + outputs].to_owned(),
    offset: 0,
    first_output,
    outputs,
  })
  values
}

///|
/// Observable allocation contract for deferred block/scalar equivalence checks.
pub fn Tape::node_count(self : Tape) -> Int32 {
  self.nodes.length()
}

///|
pub fn Tape::gradient(self : Tape, input : Value) -> Double {
  self.nodes[input.id].gradient
}

///|
pub fn Tape::sin(self : Tape, input : Value) -> Value {
  self.node(@math.sin(input.value), input.id, -1, @math.cos(input.value), 0)
}

///|
pub fn Tape::cos(self : Tape, input : Value) -> Value {
  self.node(@math.cos(input.value), input.id, -1, -@math.sin(input.value), 0)
}

///|
pub fn Tape::atan2(self : Tape, y : Value, x : Value) -> Value {
  let norm2 = x.value * x.value + y.value * y.value
  self.node(
    @math.atan2(y.value, x.value),
    y.id,
    x.id,
    x.value / norm2,
    -y.value / norm2,
  )
}

///|
pub fn Tape::acos(self : Tape, input : Value) -> Value {
  self.node(
    @math.acos(input.value),
    input.id,
    -1,
    -1.0 / (1.0 - input.value * input.value).sqrt(),
    0,
  )
}

///|
pub fn Tape::abs(self : Tape, input : Value) -> Value {
  self.node(
    input.value.abs(),
    input.id,
    -1,
    if input.value > 0 {
      1
    } else if input.value < 0 {
      -1
    } else {
      0
    },
    0,
  )
}

///|
pub fn Tape::vector(self : Tape, values : Array[Double]) -> Array[Value] {
  values.map(x => self.scalar(x))
}

///|
pub fn Tape::mean(self : Tape, values : Array[Value]) -> Value {
  guard values.length() > 0 else { abort("mean requires nonempty input") }
  let mut total = self.scalar(0)
  for v in values {
    total = self.add(total, v)
  }
  self.scale(total, 1.0 / values.length().to_double())
}
