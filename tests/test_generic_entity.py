import unittest
from uam_sim.generic_entity import GenericEntity


class TestGenericEntity(unittest.TestCase):
  """
  This class contains unit tests for the GenericEntity class.
  """

  def setUp(self):
    """
    Set up that is run before each test.
    """
    # Initialize your object here
    self.entity = GenericEntity(1, name="Test Entity", value=100)

  def test_initialization(self):
    """
    Test the initialization of the GenericEntity.
    """
    self.assertEqual(self.entity.entity_id, 1)
    self.assertEqual(self.entity.get_property('name'), "Test Entity")
    self.assertEqual(self.entity.get_property('value'), 100)

  def test_update_property(self):
    """
    Test the update_property method.
    """
    # Update property
    self.entity.update_property('value', 200)
    # Check if updated correctly
    self.assertEqual(self.entity.get_property('value'), 200)

  def tearDown(self):
    """
    Clean up that is run after each test.
    """
    del self.entity


if __name__ == '__main__':
  unittest.main()
