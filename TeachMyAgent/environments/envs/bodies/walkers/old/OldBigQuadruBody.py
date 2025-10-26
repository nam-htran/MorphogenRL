# Taken from https://github.com/flowersteam/teachDeepRL

import numpy as np
from Box2D.b2 import edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener

from TeachMyAgent.environments.envs.bodies.walkers.WalkerAbstractBody import WalkerAbstractBody
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomBodyUserData, CustomMotorUserData

HULL_POLYGONS = [
        [(-46, +13), (+6, +13), (+50, +5),
        (+50, -12), (-46, -12)]
    ]
HULL_BOTTOM_WIDTH = 96
SPEED_HIP     = 4
SPEED_KNEE    = 6

class OldBigQuadruBody(WalkerAbstractBody):
    '''
        Quadrupedal walker implemented in https://github.com/flowersteam/teachDeepRL.
    '''
    def __init__(self, scale, motors_torque=300, nb_steps_under_water=600, reset_on_hull_critical_contact=False):
        '''
            Creates a big quadrupedal walker.

            Args:
                scale: Scale value used in the environment (to adapt the embodiment to its environment)
                motors_torque: Maximum torque the embodiment can use on its motors
                 nb_steps_under_water: How many consecutive steps the embodiment can survive under water
                reset_on_hull_critical_contact: Whether a contact detected with the head should stop the episode
        '''
        super(OldBigQuadruBody, self).__init__(scale, motors_torque, nb_steps_under_water)
        self.LEG_DOWN = -8 / self.SCALE  # 0 = center of hull
        self.LEG_W, self.LEG_H = 10 / self.SCALE, 51 / self.SCALE
        self.TORQUE_PENALTY = 0.00035 / 2 # Because 2 pairs of legs
        self.reset_on_hull_critical_contact = reset_on_hull_critical_contact

        # Approximative...
        self.AGENT_WIDTH = HULL_BOTTOM_WIDTH / self.SCALE
        self.AGENT_HEIGHT = 25 / self.SCALE + \
                            self.LEG_H * 2 - self.LEG_DOWN
        self.AGENT_CENTER_HEIGHT = self.LEG_H * 2 + self.LEG_DOWN

        self.old_morphology = True

        self.nb_motors = 8
        self.state_size = self.nb_motors * 2 + 4

    def draw(self, world, init_x, init_y, force_to_center):
        HULL_FIXTURES = [
            fixtureDef(
                shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in polygon]),
                density=5.0,
                friction=0.1,
                categoryBits=0x20,
                maskBits=0x000F)
            for polygon in HULL_POLYGONS
        ]
        LEG_FD = fixtureDef(shape=polygonShape(box=(self.LEG_W / 2, self.LEG_H / 2)), density=1.0, restitution=0.0, categoryBits=0x20, maskBits=0x000F)
        LOWER_FD = fixtureDef(shape=polygonShape(box=(0.8 * self.LEG_W / 2, self.LEG_H / 2)), density=1.0, restitution=0.0, categoryBits=0x20, maskBits=0x000F)

        hull = world.CreateDynamicBody(position=(init_x, init_y), fixtures=HULL_FIXTURES)
        hull.color1, hull.color2 = (0.5, 0.4, 0.9), (0.3, 0.3, 0.5)
        hull.ApplyForceToCenter((force_to_center, 0), True)
        hull.userData = CustomBodyUserData(True, is_contact_critical=self.reset_on_hull_critical_contact, name="hull")
        self.body_parts.append(hull)
        self.reference_head_object = hull

        # ===== BẮT ĐẦU SỬA LỖI VẬT LÝ =====
        # Sử dụng logic tính toán vị trí tuyệt đối để tăng độ ổn định
        for x_anchor_rel in [-0.7, 0.5]: # Tỷ lệ vị trí mỏ neo trên thân
            absolute_x = init_x + np.interp(x_anchor_rel, [-1,1], [-HULL_BOTTOM_WIDTH / 2 / self.SCALE, HULL_BOTTOM_WIDTH / 2 / self.SCALE])
            for i in [-1, +1]:
                leg = world.CreateDynamicBody(position=(absolute_x, init_y - self.LEG_H / 2 - self.LEG_DOWN), fixtures=LEG_FD)
                leg.color1, leg.color2 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.), (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
                rjd = revoluteJointDef(bodyA=hull, bodyB=leg, anchor=(absolute_x, init_y - self.LEG_DOWN), enableMotor=True, enableLimit=True, maxMotorTorque=self.MOTORS_TORQUE, motorSpeed=i, lowerAngle=-0.8, upperAngle=1.1)
                leg.userData = CustomBodyUserData(False, name="leg")
                self.body_parts.append(leg)
                joint_motor = world.CreateJoint(rjd)
                joint_motor.userData = CustomMotorUserData(SPEED_HIP, False)
                self.motors.append(joint_motor)
                
                lower = world.CreateDynamicBody(position=(absolute_x, init_y - self.LEG_H * 3 / 2 - self.LEG_DOWN), fixtures=LOWER_FD)
                lower.color1, lower.color2 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.), (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
                rjd = revoluteJointDef(bodyA=leg, bodyB=lower, anchor=(absolute_x, init_y - self.LEG_DOWN - self.LEG_H), enableMotor=True, enableLimit=True, maxMotorTorque=self.MOTORS_TORQUE, motorSpeed=1, lowerAngle=-1.6, upperAngle=-0.1)
                lower.userData = CustomBodyUserData(True, name="lower")
                self.body_parts.append(lower)
                joint_motor = world.CreateJoint(rjd)
                joint_motor.userData = CustomMotorUserData(SPEED_KNEE, True, contact_body=lower, angle_correction=1.0)
                self.motors.append(joint_motor)