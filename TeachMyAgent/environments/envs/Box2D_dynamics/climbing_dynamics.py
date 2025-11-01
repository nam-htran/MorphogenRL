import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomUserDataObjectTypes

class ClimbingDynamics(object):
    # <<< SỬA LỖI: Thêm hàm __init__ để tạo danh sách chứa các khớp cần hủy >>>
    def __init__(self):
        self.joints_to_destroy = []
        
    def before_step_climbing_dynamics(self, actions, body, world):
        '''
        Check if sensors should grasp or release.
        If releasing and a joint exists, destroy it.
        '''
        for i in range(len(body.sensors)):
            action_to_check = actions[len(actions) - i - 1]
            sensor_to_check = body.sensors[len(body.sensors) - i - 1]

            if action_to_check > 0:
                sensor_to_check.userData.ready_to_attach = True
            else:
                sensor_to_check.userData.ready_to_attach = False
                if sensor_to_check.userData.has_joint:
                    sensor_to_check.userData.has_joint = False
                    joint_to_destroy = next((_joint.joint for _joint in sensor_to_check.joints
                                             if isinstance(_joint.joint, Box2D.b2RevoluteJoint)), None)
                    if joint_to_destroy is not None:
                        ## <<< SỬA LỖI: Trì hoãn việc hủy khớp thay vì hủy ngay lập tức >>>
                        # Dòng cũ gây lỗi: world.DestroyJoint(joint_to_destroy)
                        self.joints_to_destroy.append(joint_to_destroy)

    def after_step_climbing_dynamics(self, contact_detector, world):
        '''
        Create climbing joints if sensors are still overlapping after Box2D solver execution.
        '''
        ## <<< SỬA LỖI: Phá hủy các khớp đã được đánh dấu một cách an toàn >>>
        # Việc này được thực hiện sau world.Step(), nên an toàn.
        for joint in self.joints_to_destroy:
            world.DestroyJoint(joint)
        self.joints_to_destroy.clear() # Dọn dẹp danh sách
        
        for sensor in contact_detector.contact_dictionaries:
            if len(contact_detector.contact_dictionaries[sensor]) > 0 and \
                    sensor.userData.ready_to_attach and not sensor.userData.has_joint:
                other_body = contact_detector.contact_dictionaries[sensor][0]

                # Simple overlap check (fast approximation)
                other_body_shape = other_body.fixtures[0].shape
                x_values = [v[0] for v in other_body_shape.vertices]
                y_values = [v[1] for v in other_body_shape.vertices]
                radius = sensor.fixtures[0].shape.radius + 0.01

                if (sensor.worldCenter[0] + radius > min(x_values) and sensor.worldCenter[0] - radius < max(x_values) and
                    sensor.worldCenter[1] + radius > min(y_values) and sensor.worldCenter[1] - radius < max(y_values)):
                    rjd = revoluteJointDef(
                        bodyA=sensor,
                        bodyB=other_body,
                        anchor=sensor.worldCenter
                    )

                    joint = world.CreateJoint(rjd)
                    joint.bodyA.userData.joint = joint
                    sensor.userData.has_joint = True
                else:
                    contact_detector.contact_dictionaries[sensor].remove(other_body)
                    if len(contact_detector.contact_dictionaries[sensor]) == 0:
                        sensor.userData.has_contact = False
class ClimbingContactDetector(contactListener):
    '''
    Stores contacts between sensors and graspable surfaces.
    '''
    def __init__(self):
        super(ClimbingContactDetector, self).__init__()
        self.contact_dictionaries = {}

    def BeginContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        for idx, body in enumerate(bodies):
            if body.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR and body.userData.check_contact:
                other_body = bodies[(idx + 1) % 2]
                if other_body.userData.object_type in (
                    CustomUserDataObjectTypes.GRIP_TERRAIN,
                    CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN,
                ):
                    body.userData.has_contact = True
                    if body in self.contact_dictionaries:
                        self.contact_dictionaries[body].append(other_body)
                    else:
                        self.contact_dictionaries[body] = [other_body]

    def EndContact(self, contact):
        fA, fB = contact.fixtureA, contact.fixtureB

        if not (hasattr(fA, 'body') and hasattr(fB, 'body') and fA.body and fB.body and
                hasattr(fA.body, 'userData') and hasattr(fB.body, 'userData') and
                fA.body.userData and fB.body.userData):
            return

        bodies = [fA.body, fB.body]
        for idx, body in enumerate(bodies):
            if (body.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR and
                body.userData.check_contact and body.userData.has_contact):

                other_body = bodies[(idx + 1) % 2]

                if hasattr(other_body, 'userData') and other_body.userData is not None:
                    if body in self.contact_dictionaries and other_body in self.contact_dictionaries[body]:
                        self.contact_dictionaries[body].remove(other_body)

                    if body in self.contact_dictionaries and len(self.contact_dictionaries[body]) == 0:
                        body.userData.has_contact = False

    def Reset(self):
        self.contact_dictionaries = {}