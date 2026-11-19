///|
test "RNG snapshots restore exact stream without seed-zero aliasing or shared state" {
  let zero = @numeric.Rng::new(0U)
  let one = @numeric.Rng::new(1U)
  assert_true(zero.state() != one.state())
  let snapshot = zero.snapshot()
  let state = zero.state()
  let first = zero.normal()
  let next = zero.uniform()
  assert_eq(snapshot.normal(), first)
  assert_eq(snapshot.uniform(), next)
  zero.set_state(state)
  assert_eq(zero.normal(), first)
  assert_eq(zero.uniform(), next)
  let saved = one.state()
  ignore(zero.normal())
  assert_eq(one.state(), saved)
}
