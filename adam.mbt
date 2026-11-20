///|
/// Persistent Adam moments and bias-correction step.
pub struct Adam {
  first : Array[Double]
  second : Array[Double]
  mut steps : Int32
  mut learning_rate : Double
  beta1 : Double
  beta2 : Double
  epsilon : Double
} derive(ToJson, FromJson)

///|
pub fn Adam::new(
  count : Int32,
  learning_rate : Double,
  beta1~ : Double = 0.9,
  beta2~ : Double = 0.999,
  epsilon~ : Double = 0.00000001,
) -> Adam {
  guard count >= 0 &&
    finite(learning_rate) &&
    learning_rate >= 0 &&
    beta1 >= 0 &&
    beta1 < 1 &&
    beta2 >= 0 &&
    beta2 < 1 &&
    epsilon > 0 &&
    finite(epsilon) else {
    abort("invalid Adam configuration")
  }
  {
    first: Array::make(count, 0.0),
    second: Array::make(count, 0.0),
    steps: 0,
    learning_rate,
    beta1,
    beta2,
    epsilon,
  }
}

///|
/// Return pre-clipping norm, update only explicitly supplied parameters.
pub fn Adam::step(
  self : Adam,
  parameters : Array[Double],
  tape : Tape,
  bound : Array[Value],
  max_norm~ : Double = 1.0,
  max_value~ : Double = 1.0 / 0.0,
  gradient_trace? : (Array[Double], Array[Double], Double) -> Unit,
) -> Double {
  guard parameters.length() == self.first.length() &&
    bound.length() == parameters.length() else {
    abort("Adam shape mismatch")
  }
  // Check restored/decoded counters before callbacks or optimizer state changes.
  guard self.steps >= 0 && self.steps < 2147483647 else {
    abort("Adam step counter is negative or exhausted")
  }
  let gradient = bound.map(v => tape.gradient(v))
  let mut sum = 0.0
  for g in gradient {
    guard finite(g) else { abort("nonfinite gradient") }
    sum += g * g
  }
  let norm = sum.sqrt()
  guard finite(norm) && max_norm >= 0 && max_value >= 0 else {
    abort("nonfinite gradient norm or invalid clipping bound")
  }
  let coefficient = @cmp.minimum(1.0, max_norm / (norm + 0.000001))
  let applied = gradient.map(g => {
    let raw = g * coefficient
    if raw > max_value {
      max_value
    } else if raw < -max_value {
      -max_value
    } else {
      raw
    }
  })
  if gradient_trace is Some(capture) {
    capture(gradient.copy(), applied.copy(), norm)
  }
  self.steps += 1
  let correction1 = 1.0 - @math.pow(self.beta1, self.steps.to_double())
  let correction2 = 1.0 - @math.pow(self.beta2, self.steps.to_double())
  for i in 0..<parameters.length() {
    let g = applied[i]
    self.first[i] = self.beta1 * self.first[i] + (1 - self.beta1) * g
    self.second[i] = self.beta2 * self.second[i] + (1 - self.beta2) * g * g
    parameters[i] -= self.learning_rate *
      (self.first[i] / correction1) /
      ((self.second[i] / correction2).sqrt() + self.epsilon)
  }
  norm
}

///|
pub fn Adam::set_learning_rate(self : Adam, learning_rate : Double) -> Unit {
  guard learning_rate >= 0 && finite(learning_rate) else {
    abort("invalid Adam learning rate")
  }
  self.learning_rate = learning_rate
}

///|
pub fn Adam::snapshot(self : Adam) -> Adam {
  {
    first: self.first.copy(),
    second: self.second.copy(),
    steps: self.steps,
    learning_rate: self.learning_rate,
    beta1: self.beta1,
    beta2: self.beta2,
    epsilon: self.epsilon,
  }
}

///|
pub fn Adam::restore(self : Adam, saved : Adam) -> Unit {
  guard self.first.length() == saved.first.length() &&
    self.beta1 == saved.beta1 &&
    self.beta2 == saved.beta2 &&
    self.epsilon == saved.epsilon else {
    abort("incompatible Adam state")
  }
  for i in 0..<self.first.length() {
    self.first[i] = saved.first[i]
    self.second[i] = saved.second[i]
  }
  self.steps = saved.steps
  self.learning_rate = saved.learning_rate
}
