///|
test "scaled pivot solve handles row swaps multiple RHS and preserves inputs" {
  let a = [[0.0, 2.0], [1.0, 3.0]]
  let b = [[4.0, 2.0], [7.0, 4.0]]
  assert_eq(@linalg.solve(a, b), [[1.0, 1.0], [2.0, 1.0]])
  assert_eq(a, [[0.0, 2.0], [1.0, 3.0]])
  assert_eq(b, [[4.0, 2.0], [7.0, 4.0]])
  assert_true(
    (try? @linalg.solve([[1.0, 2.0], [2.0, 4.0]], [[1.0], [2.0]])) is Err(_),
  )
}

///|
test "pivoted Householder least squares recovers multichannel coefficients" {
  let a = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
  let b = [[2.0, -1.0], [5.0, 1.0], [8.0, 3.0], [11.0, 5.0]]
  let result = @linalg.least_squares(a, b)
  assert_eq(result.rank, 2)
  let expected = [[2.0, -1.0], [3.0, 2.0]]
  for i in 0..<2 {
    for j in 0..<2 {
      assert_true(
        (result.solution[i][j] - expected[i][j]).abs() < 0.000000000001,
      )
    }
  }
  assert_true(result.residual_sum_squares.all(x => x < 0.00000000000000000001))
  assert_true(result.pivots[0] == 1)
}

///|
test "QR rank deficiency returns explicit rank and valid basic solution" {
  let a = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
  let result = @linalg.least_squares(a, [[3.0], [6.0], [9.0]])
  assert_eq(result.rank, 1)
  assert_true(result.residual_sum_squares[0] < 0.00000000000000000001)
  assert_true(result.solution[0][0] == 0 || result.solution[1][0] == 0)
  let wide = @linalg.least_squares([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], [
    [2.0],
    [3.0],
  ])
  assert_eq(wide.rank, 2)
  assert_true(wide.residual_sum_squares[0] < 0.00000000000000000001)
  assert_true((try? @linalg.least_squares([[1.0], [2.0]], [[1.0]])) is Err(_))
}

///|
test "Householder scaling avoids overflow from raw squared column norms" {
  let result = @linalg.least_squares(
    [[1.0e150, 0.0], [0.0, 1.0e150], [1.0e150, 1.0e150]],
    [[1.0e150], [2.0e150], [3.0e150]],
  )
  assert_eq(result.rank, 2)
  assert_true((result.solution[0][0] - 1).abs() < 0.000000000001)
  assert_true((result.solution[1][0] - 2).abs() < 0.000000000001)
}

///|
test "QR rejects finite inputs with unrepresentable norms instead of reporting zero rank" {
  // sqrt(2)*1.7e308 exceeds Double maximum despite finite individual entries.
  assert_true(
    (try? @linalg.least_squares([[1.7e308], [1.7e308]], [[0.0], [0.0]]))
    is Err(_),
  )
  assert_true(
    (try? @linalg.least_squares([[1.0, 1.7e308], [0.0, 1.7e308]], [[0.0], [0.0]]))
    is Err(_),
  )
  assert_true(
    (try? @linalg.least_squares([[1.7e308], [1.7e308]], [[0.0], [0.0]], rcond=0))
    is Err(_),
  )
}

///|
test "QR rejects overflowing reflections even when initial norms are finite" {
  assert_true(
    (try? @linalg.least_squares([[1.0e308], [0.0]], [[0.0], [0.0]])) is Err(_),
  )
  assert_true(
    (try? @linalg.least_squares([[1.0], [0.0]], [[1.0e308], [0.0]])) is Err(_),
  )
}
