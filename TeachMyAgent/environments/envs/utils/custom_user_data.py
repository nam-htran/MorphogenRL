from enum import Enum


class CustomUserDataObjectTypes(Enum):
    """Enumeration of object types for custom user data."""
    BODY_OBJECT = 0
    WATER = 1
    TERRAIN = 2
    GRIP_TERRAIN = 3       # Graspable terrain
    MOTOR = 4
    BODY_SENSOR = 5
    SENSOR_GRIP_TERRAIN = 6  # Graspable sensor terrain (e.g., creeper)


class CustomUserData:
    """
    Base class for storing custom properties on simulation objects.
    """
    def __init__(self, name: str, object_type: CustomUserDataObjectTypes):
        self.name = name
        self.object_type = object_type


class CustomMotorUserData(CustomUserData):
    """
    User data for joints with `enableMotor=True`.
    Stores motor control parameters and contact linkage info.
    """
    def __init__(self, speed_control: bool, check_contact: bool,
                 angle_correction: float = 0.0, contact_body=None):
        """
        Args:
            speed_control: Whether this motor is controlled by speed.
            check_contact: If True, a `contact_body` must be provided.
                           Used in the observation space to provide contact info
                           for linked objects.
            angle_correction: Correction term for the joint angle (default=0.0).
            contact_body: Body to monitor for contact events.
        """
        super().__init__("motor", CustomUserDataObjectTypes.MOTOR)
        self.speed_control = speed_control
        self.check_contact = check_contact
        self.angle_correction = angle_correction
        self.contact_body = contact_body


class CustomBodyUserData(CustomUserData):
    """
    User data for body parts.
    Defines collision and termination behavior.
    """
    def __init__(self, check_contact: bool, is_contact_critical: bool = False,
                 name: str = "body_part",
                 object_type: CustomUserDataObjectTypes = CustomUserDataObjectTypes.BODY_OBJECT):
        """
        Args:
            check_contact: If False, collisions for this body are ignored.
            is_contact_critical: If True, a collision ends the episode.
            name: Name of the body part.
            object_type: Type of body (default = BODY_OBJECT).
        """
        super().__init__(name, object_type)
        self.check_contact = check_contact
        self.is_contact_critical = is_contact_critical
        self.has_contact = False


class CustomBodySensorUserData(CustomBodyUserData):
    """
    User data for sensors attached to bodies.
    """
    def __init__(self, check_contact: bool, is_contact_critical: bool = False,
                 name: str = "body_sensor"):
        super().__init__(
            check_contact=check_contact,
            is_contact_critical=is_contact_critical,
            name=name,
            object_type=CustomUserDataObjectTypes.BODY_SENSOR
        )
        self.has_joint = False
        self.ready_to_attach = False
