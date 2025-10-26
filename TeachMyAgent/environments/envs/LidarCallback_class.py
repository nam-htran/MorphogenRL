# TeachMyAgent/environments/envs/LidarCallback_class.py
import Box2D
from .utils.custom_user_data import CustomUserDataObjectTypes

class LidarCallback(Box2D.b2.rayCastCallback):
    '''
        Callback function triggered when lidar detects an object.
    '''
    def __init__(self, agent_mask_filter):
        Box2D.b2.rayCastCallback.__init__(self)
        self.agent_mask_filter = agent_mask_filter
        self.fixture = None
        self.is_water_detected = False
        self.is_creeper_detected = False
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & self.agent_mask_filter) == 0:
            return -1
        self.p2 = point
        self.fraction = fraction
        self.is_water_detected = True if fixture.body.userData.object_type == CustomUserDataObjectTypes.WATER else False
        self.is_creeper_detected = True if fixture.body.userData.object_type == CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN else False
        return fraction