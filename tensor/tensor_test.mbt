///|
test "strided views share storage while clone and noncontiguous reshape copy" {
  let a = @tensor.Tensor::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3])
  let t = a.transpose(0, 1)
  assert_eq(t.shape(), [3, 2])
  assert_eq(t.strides(), [1, 3])
  assert_eq(t.to_array(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
  t.set([1, 0], 10)
  assert_eq(a.get([0, 1]), 10)
  let cloned = t.clone()
  cloned.set([0, 0], 99)
  assert_eq(a.get([0, 0]), 0)
  assert_true(
    try t.view([6]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let flat = t.reshape([6])
  flat.set([0], 88)
  assert_eq(a.get([0, 0]), 0)
  let reversed = a.slice(1, -1, 3, step=-1)
  assert_eq(reversed.to_array(), [2.0, 10.0, 0.0, 5.0, 4.0, 3.0])
  assert_eq(reversed.strides(), [3, -1])
}

///|
test "broadcasting reductions and vector batched matmul preserve explicit shapes" {
  let a = @tensor.Tensor::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let bias = @tensor.Tensor::from_array([10.0, 20.0, 30.0], [3])
  assert_eq(a.add(bias).to_array(), [11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
  assert_eq(a.sum(axis=0).to_array(), [5.0, 7.0, 9.0])
  assert_eq(a.mean(axis=1, keepdim=true).shape(), [2, 1])
  assert_eq(a.mean(axis=1).to_array(), [2.0, 5.0])
  let vector = @tensor.Tensor::from_array([1.0, 2.0, 3.0], [3])
  assert_eq(a.matmul(vector).to_array(), [14.0, 32.0])
  assert_eq(vector.matmul(vector).shape(), [])
  assert_eq(vector.matmul(vector).get([]), 14)
  let matrix = @tensor.Tensor::from_array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2])
  let batched = a.reshape([1, 2, 3]).broadcast_to([4, 2, 3]).matmul(matrix)
  assert_eq(batched.shape(), [4, 2, 2])
  assert_eq(batched.to_array(), [
    4.0, 5.0, 10.0, 11.0, 4.0, 5.0, 10.0, 11.0, 4.0, 5.0, 10.0, 11.0, 4.0, 5.0, 10.0,
    11.0,
  ])
}

///|
test "F32 stores actual Float and rounds each ordered accumulation" {
  let a = @tensor.Tensor::from_array(
    [16777216.0, 1.0, -16777216.0],
    [3],
    dtype=F32,
  )
  assert_eq(a.sum().get([]), 0)
  let double = @tensor.Tensor::from_array(
    [16777216.0, 1.0, -16777216.0],
    [3],
    dtype=F64,
  )
  assert_eq(double.sum().get([]), 1)
  assert_eq(a.matmul(@tensor.Tensor::full([3], 1, dtype=F32)).get([]), 0)
  let stored = @tensor.Tensor::from_array([16777217.0], [1], dtype=F32)
  assert_eq(stored.get([0]), 16777216)
  assert_true(
    try a.add(double) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}

///|
test "empty tensors and invalid views have explicit behavior" {
  let empty = @tensor.Tensor::zeros([2, 0, 3])
  assert_eq(empty.numel(), 0)
  assert_eq(empty.sum().get([]), 0)
  let huge_empty = @tensor.Tensor::zeros([0, 2147483647, 2147483647])
  assert_eq(huge_empty.numel(), 0)
  assert_eq(empty.matmul(@tensor.Tensor::zeros([3, 4])).shape(), [2, 0, 4])
  assert_true(
    try @tensor.Tensor::zeros([-1]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try empty.broadcast_to([2, 1, 3]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let scalar = @tensor.Tensor::scalar(2)
  let broadcast = scalar.broadcast_to([2, 2])
  broadcast.set([1, 0], 3)
  assert_eq(scalar.get([]), 3)
}
