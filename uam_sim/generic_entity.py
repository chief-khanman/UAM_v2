class GenericEntity:
  """
  A generic class to represent different entities in a UAM simulation environment.

  Attributes:
      entity_id (int): Unique identifier for the entity.
      properties (dict): Dictionary to hold various attributes of the entity.
  """

  def __init__(self, entity_id, **properties):
    """
    Initializes a new Generic Entity object with the given ID and properties.

    Args:
        entity_id (int): Unique identifier for the entity.
        **properties (dict): Arbitrary number of properties as key-value pairs.
    """
    self.entity_id = entity_id
    self.properties = properties

  def update_property(self, key, value):
    """
    Updates or adds a property of the entity.

    Args:
        key (str): The property key.
        value (any): The new value for the property.
    """
    self.properties[key] = value

  def get_property(self, key):
    """
    Retrieves the value of a property.

    Args:
        key (str): The property key.

    Returns:
        The value of the specified property, or None if the key does not exist.
    """
    return self.properties.get(key, None)

  def __str__(self):
    """
    Returns a string representation of the entity, showing its ID and properties.

    Returns:
        str: A string representation of the entity.
    """
    properties_str = ', '.join([f"{k}: {v}" for k, v in self.properties.items()])
    return f"Entity {self.entity_id}: {properties_str}"
