///|
/// Scalar reference deliberately retains the former two-nodes-per-MAC implementation for deferred differential checks.
fn scalar_forward(
  model : @numeric.MLP,
  tape : @numeric.Tape,
  parameters : Array[@numeric.Value],
  input : Array[@numeric.Value],
) -> Array[@numeric.Value] {
  let mut values = input
  let mut offset = 0
  for layer in 0..<(model.dimensions.length() - 1) {
    let nin = model.dimensions[layer]
    let nout = model.dimensions[layer + 1]
    values = Array::makei(nout, row => {
      let mut sum = parameters[offset + nin * nout + row]
      for column in 0..<nin {
        sum = tape.add(
          sum,
          tape.mul(parameters[offset + row * nin + column], values[column]),
        )
      }
      if layer < model.dimensions.length() - 2 {
        tape.silu(sum)
      } else {
        sum
      }
    })
    offset += nin * nout + nout
  }
  values
}

///|
test "dense block matches scalar forward gradients and direct no-tape inference" {
  let model = @numeric.MLP::new([3, 4, 2], @numeric.Rng::new(321U))
  let inputs = [0.3, -0.7, 0.2]
  let block = @numeric.Tape::new()
  let bp = model.bind(block)
  let bi = block.vector(inputs)
  let bo = model.forward(block, bp, bi)
  let scalar = @numeric.Tape::new()
  let sp = model.bind(scalar)
  let si = scalar.vector(inputs)
  let so = scalar_forward(model, scalar, sp, si)
  assert_eq(bo.map(x => x.value), so.map(x => x.value))
  assert_eq(model.predict(inputs), bo.map(x => x.value))
  let block_objective = block.add(
    block.mul(bo[0], bo[0]),
    block.scale(bo[1], 0.7),
  )
  let scalar_objective = scalar.add(
    scalar.mul(so[0], so[0]),
    scalar.scale(so[1], 0.7),
  )
  block.backward(block_objective)
  scalar.backward(scalar_objective)
  for i in 0..<bp.length() {
    assert_true(
      (block.gradient(bp[i]) - scalar.gradient(sp[i])).abs() < 0.000000000001,
    )
  }
  for i in 0..<bi.length() {
    assert_true(
      (block.gradient(bi[i]) - scalar.gradient(si[i])).abs() < 0.000000000001,
    )
  }
  assert_true(
    block.node_count() - bp.length() - bi.length() <
    (scalar.node_count() - sp.length() - si.length()) / 2,
  )
  // Verify parameters retain their existing exact flattened checkpoint order.
  let snapshot = model.snapshot()
  assert_eq(snapshot.parameters, model.parameters)
  assert_eq(snapshot.dimensions, model.dimensions)
  snapshot.parameters[0] += 1
  assert_true(snapshot.parameters != model.parameters)
}

///|
test "dense reverse kernel handles shared inputs disconnected outputs and repeated backward" {
  let tape = @numeric.Tape::new()
  let x = tape.scalar(0.4)
  let parameters = tape.vector([2.0, 3.0, 4.0, 5.0, 0.1, 0.2])
  let output = tape.linear(parameters, [x, x], 2)
  tape.backward(output[0])
  assert_true((tape.gradient(x) - 5).abs() < 0.000000000001)
  assert_eq(tape.gradient(parameters[2]), 0)
  tape.backward(output[1])
  assert_true((tape.gradient(x) - 9).abs() < 0.000000000001)
  assert_eq(tape.gradient(parameters[0]), 0)
  let both = tape.add(output[0], output[1])
  tape.backward(both)
  assert_true((tape.gradient(x) - 14).abs() < 0.000000000001)
}

///|
test "recurrent dense unroll input gradient matches central difference" {
  let model = @numeric.MLP::new([2, 3, 2], @numeric.Rng::new(17U))
  let tape = @numeric.Tape::new()
  let parameters = model.bind(tape)
  let initial = tape.vector([0.2, -0.1])
  let mut state = initial
  for _ in 0..<8 {
    state = model.forward(tape, parameters, state)
  }
  tape.backward(tape.add(state[0], tape.scale(state[1], 0.3)))
  let evaluate = fn(x : Double) -> Double {
    let mut values = [x, -0.1]
    for _ in 0..<8 {
      values = model.predict(values)
    }
    values[0] + 0.3 * values[1]
  }
  let numeric = (evaluate(0.200001) - evaluate(0.199999)) / 0.000002
  assert_true((tape.gradient(initial[0]) - numeric).abs() < 0.000000001)
  assert_true(tape.gradient(initial[0]).abs() > 0)
}

///|
test "optimizer trace reports actual raw and clipped gradients without aliasing" {
  let tape = @numeric.Tape::new()
  let parameters = [1.0, -2.0]
  let bound = tape.vector(parameters)
  tape.backward(tape.add(tape.scale(bound[0], 3), tape.scale(bound[1], 4)))
  let captured : Array[Array[Double]] = []
  let captured_norm : Array[Double] = []
  let optimizer = @numeric.Adam::new(2, 0.01)
  let norm = optimizer.step(
    parameters,
    tape,
    bound,
    max_norm=1,
    max_value=0.1,
    gradient_trace=(raw, applied, norm) => {
      captured_norm.push(norm)
      captured.push(raw.copy())
      captured.push(applied.copy())
      raw[0] = 999
      applied[0] = 999
    },
  )
  assert_eq(norm, 5)
  assert_eq(captured_norm, [5.0])
  assert_eq(captured, [[3.0, 4.0], [0.1, 0.1]])
  assert_true((parameters[0] - 0.9900000009999999).abs() < 0.000000000001)
  assert_true((parameters[1] + 2.009999999).abs() < 0.000000000001)
}
