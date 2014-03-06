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
