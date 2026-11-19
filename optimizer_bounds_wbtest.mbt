///|
test "panic Adam rejects exhausted restored counters before an update" {
  let optimizer = Adam::new(1, 0.01)
  let restored = optimizer.snapshot()
  restored.steps = 2147483647
  optimizer.restore(restored)
  let tape = Tape::new()
  let parameters = [1.0]
  let bound = tape.vector(parameters)
  tape.backward(bound[0])
  ignore(optimizer.step(parameters, tape, bound))
}

///|
test "panic Adam rejects negative decoded counters before an update" {
  let optimizer : Adam = @json.from_json({
    "first": [0.0],
    "second": [0.0],
    "steps": -1,
    "learning_rate": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 0.00000001,
  })
  let tape = Tape::new()
  let parameters = [1.0]
  let bound = tape.vector(parameters)
  tape.backward(bound[0])
  ignore(optimizer.step(parameters, tape, bound))
}

///|
test "Adam accepts its final representable increment and traces before state changes" {
  let optimizer = Adam::new(1, 0.01)
  optimizer.steps = 2147483646
  let tape = Tape::new()
  let parameters = [1.0]
  let bound = tape.vector(parameters)
  tape.backward(bound[0])
  let observed : Array[(Int32, Double, Double)] = []
  ignore(
    optimizer.step(parameters, tape, bound, gradient_trace=(_, _, _) => {
      observed.push((optimizer.steps, optimizer.first[0], parameters[0]))
    }),
  )
  assert_eq(observed, [(2147483646, 0.0, 1.0)])
  assert_eq(optimizer.steps, 2147483647)
  assert_true(finite(parameters[0]))
  assert_true(parameters[0] < 1)
}
