def test_visibility_descriptor_fields():
  from nose.tools import assert_equal
  from operator import itemgetter
  from purify import visibility_descriptor as vd

  fields = [(name, offset, dtype) for name, (dtype, offset) in vd.fields.iteritems()]
  fields = sorted(fields, key=itemgetter(1))

  # Check field names are in right order, e.g equal to C 
  names = [name for name, offset, dtype in fields]
  assert_equal(names, ['u', 'v', 'w', 'noise', 'measurement']) 

  # Check that fields are the right kind
  kinds = [dtype.kind for name, offset, dtype in fields]
  assert_equal(kinds, ['f'] * 3 + ['c'] * 2)

def test_if_a_visibility():
  """ Python-created object is, or isn't a visibility """
  from nose.tools import assert_true, assert_false
  from numpy import array, zeros
  from purify._purify import _is_a_visibility
  from purify import visibility_descriptor as vd

  invalid_inputs = [
    0, array(1), array([0, 0, 0, 0, 0], dtype=int),
    array([0, 0, 0, 0, 0], dtype=float),
    array([0, 0, 0, 0, 0], dtype=complex),
    # Following input fails because it is of the wrong type
    # It can be transformed to a visiblity. But we are checking for exact type.
    [ (0, 0, 0, 0, 0), 
      (0.1, 0.2, 0.3, 0.4 + 0.5j, 0.6 + 0.7j), 
      (0, 0, 0, 0, 0), (0, 0, 0, 0, 0) ]
  ]
  for input in invalid_inputs:
    message = "Shoud not be a visibility %r" % input
    assert_false( _is_a_visibility(input), message )
  
  valid_inputs = [
    array([ (0, 0, 0, 0, 0), 
            (0.1, 0.2, 0.3, 0.4 + 0.5j, 0.6 + 0.7j), 
            (0, 0, 0, 0, 0), (0, 0, 0, 0, 0) ], dtype=vd),
    zeros((2, 3), dtype=vd)
  ]
  for input in valid_inputs: 
    message = "Shoud be a visibility %r" % input
    assert_true(_is_a_visibility(input));

def test_visibility_memory_layout():
  from numpy import array
  from numpy.testing import assert_allclose
  from purify._purify import _flip_element_sign
  from purify import visibility_descriptor as vd

  actual = array([ (0, 0, 0, 0, 0), 
                   (0.1, 0.2, 0.3, 0.4 + 0.5j, 0.6 + 0.7j), 
                   (0, 0, 0, 0, 0), (0, 0, 0, 0, 0) ], dtype=vd)
  expected = actual.copy()

  def check_equal(actual, expected):
    for name in expected.dtype.fields.iterkeys():
      assert_allclose(actual[name], expected[name])

  # Check u, v, w
  for i in range(3):
    expected[1][i] = -expected[1][i]
    _flip_element_sign(actual, 1, i)
    check_equal(actual, expected)

  # Check noise, measurement
  for i in range(2):
    # For simplicity, flip thinks everything is a real. Complex numbers are two reals.
    # So this test checks that real and imaginary part are in expected order, and the location in
    # memory
    # Checking flipping real part first
    _flip_element_sign(actual, 1, 3 + 2 * i)
    expected[1][i + 3] = -expected[1][i + 3].conjugate()
    check_equal(actual, expected)

    # Checking flipping imaginary part second
    _flip_element_sign(actual, 1, 3 + 2 * i + 1)
    expected[1][i + 3] = expected[1][i + 3].conjugate()
    check_equal(actual, expected)
