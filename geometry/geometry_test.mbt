///|
test "canonical geometry preserves copied vector and transform contracts" {
  let x = @geometry.vec3(1, 0, 0)
  let y = @geometry.vec3(0, 1, 0)
  assert_eq(x.cross(y), @geometry.vec3(0, 0, 1))
  assert_eq(x.dot(y), 0)
  let parent = @geometry.transform3(
    @geometry.identity_basis3(),
    @geometry.vec3(1, 2, 3),
  )
  let child = @geometry.transform3(
    @geometry.axis_angle_basis3(@geometry.vec3(0, 0, 1), 0.3),
    @geometry.vec3(4, 5, 6),
  )
  let point = @geometry.vec3(0.2, -0.1, 0.5)
  let composed = parent.compose(child).transform_point(point)
  let sequential = parent.transform_point(child.transform_point(point))
  assert_true((composed.x - sequential.x).abs() < 0.000000000001)
  assert_true((composed.y - sequential.y).abs() < 0.000000000001)
  assert_true((composed.z - sequential.z).abs() < 0.000000000001)
  assert_eq(
    @geometry.normalized_axis(@geometry.vec3(0, 0, 0)),
    @geometry.vec3(0, 0, 1),
  )
}
