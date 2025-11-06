# TeachMyAgent/environments/envs/Box2D_dynamics/water_dynamics.py
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from copy import copy
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomUserDataObjectTypes
import numpy as np


class WaterDynamics(object):
    '''
    Simplified water physics simulation using buoyancy, drag, lift, and push forces.
    Based on: https://www.iforce2d.net/b2dtut/buoyancy
    '''
    def __init__(self, gravity, drag_mod=0.25, lift_mod=0.25, push_mod=0.05,
                 max_drag=2000, max_lift=500, max_push=20):
        self.gravity = gravity
        self.drag_mod = drag_mod
        self.lift_mod = lift_mod
        self.push_mod = push_mod
        self.max_drag = max_drag
        self.max_lift = max_lift
        self.max_push = max_push

    def compute_centroids(self, vectors):
        '''Compute the centroid and area of a polygon.'''
        count = len(vectors)
        assert count >= 3

        c = Box2D.b2Vec2(0, 0)
        area = 0
        ref_point = Box2D.b2Vec2(0, 0)
        inv3 = 1 / 3

        for i in range(count):
            p1 = ref_point
            p2 = vectors[i]
            p3 = vectors[i + 1] if i + 1 < count else vectors[0]

            e1 = p2 - p1
            e2 = p3 - p1
            d = Box2D.b2Cross(e1, e2)

            triangle_area = 0.5 * d
            area += triangle_area
            c += triangle_area * inv3 * (p1 + p2 + p3)

        if area > Box2D.b2_epsilon:
            c *= 1 / area
        else:
            area = 0

        return c, area

    def inside(self, cp1, cp2, p):
        '''Check if point p is inside an edge defined by cp1, cp2.'''
        return (cp2.x - cp1.x) * (p.y - cp1.y) > (cp2.y - cp1.y) * (p.x - cp1.x)

    def intersection(self, cp1, cp2, s, e):
        '''Find the intersection point of two line segments.'''
        dc = Box2D.b2Vec2(cp1.x - cp2.x, cp1.y - cp2.y)
        dp = Box2D.b2Vec2(s.x - e.x, s.y - e.y)
        n1 = cp1.x * cp2.y - cp1.y * cp2.x
        n2 = s.x * e.y - s.y * e.x
        n3 = 1.0 / (dc.x * dp.y - dc.y * dp.x)
        return Box2D.b2Vec2((n1 * dp.x - n2 * dc.x) * n3,
                            (n1 * dp.y - n2 * dc.y) * n3)

    def find_intersection(self, fixture_A, fixture_B):
        '''Find intersection polygon between two fixtures.'''
        output_vertices = []
        polygon_A = fixture_A.shape
        polygon_B = fixture_B.shape

        for v in polygon_A.vertices:
            output_vertices.append(fixture_A.body.GetWorldPoint(v))

        clip_polygon = [fixture_B.body.GetWorldPoint(v) for v in polygon_B.vertices]

        cp1 = clip_polygon[-1]
        for cp2 in clip_polygon:
            if not output_vertices:
                break

            input_list = copy(output_vertices)
            output_vertices.clear()
            s = input_list[-1]

            for e in input_list:
                if self.inside(cp1, cp2, e):
                    if not self.inside(cp1, cp2, s):
                        output_vertices.append(self.intersection(cp1, cp2, s, e))
                    output_vertices.append(e)
                elif self.inside(cp1, cp2, s):
                    output_vertices.append(self.intersection(cp1, cp2, s, e))
                s = e
            cp1 = cp2
        return len(output_vertices) != 0, output_vertices

    def calculate_forces(self, fixture_pairs):
        '''Apply buoyancy, drag, lift, and push forces for intersecting fixtures.'''
        for pair in fixture_pairs:
            density = pair[0].density
            has_intersection, intersection_points = self.find_intersection(pair[0], pair[1])

            if not has_intersection:
                continue

            centroid, area = self.compute_centroids(intersection_points)

            # Buoyancy
            displaced_mass = density * area
            buoyancy_force = displaced_mass * -self.gravity
            pair[1].body.ApplyForce(force=buoyancy_force, point=centroid, wake=True)

            # Hydrodynamic forces
            for i in range(len(intersection_points)):
                v0 = intersection_points[i]
                v1 = intersection_points[(i + 1) % len(intersection_points)]
                mid_point = 0.5 * (v0 + v1)

                # Drag
                vel_dir = pair[1].body.GetLinearVelocityFromWorldPoint(mid_point) - \
                          pair[0].body.GetLinearVelocityFromWorldPoint(mid_point)
                vel = vel_dir.Normalize()

                edge = v1 - v0
                edge_length = edge.Normalize()
                normal = Box2D.b2Cross(-1, edge)
                drag_dot = Box2D.b2Dot(normal, vel_dir)

                if drag_dot >= 0:  # Backward edge
                    drag_mag = drag_dot * self.drag_mod * edge_length * density * vel * vel
                    drag_mag = min(drag_mag, self.max_drag)
                    drag_force = drag_mag * -vel_dir
                    pair[1].body.ApplyForce(force=drag_force, point=mid_point, wake=True)

                    # Lift
                    lift_dot = Box2D.b2Dot(edge, vel_dir)
                    lift_mag = drag_dot * lift_dot * self.lift_mod * edge_length * density * vel * vel
                    lift_mag = min(lift_mag, self.max_lift)
                    lift_dir = Box2D.b2Cross(1, vel_dir)
                    lift_force = lift_mag * lift_dir
                    pair[1].body.ApplyForce(force=lift_force, point=mid_point, wake=True)

                # Push (torque-based linear force)
                body_to_check = pair[1].body
                joints_to_check = [
                    joint_edge.joint for joint_edge in body_to_check.joints
                    if joint_edge.joint.bodyB == body_to_check
                ]

                for joint in joints_to_check:
                    if joint.lowerLimit < joint.angle < joint.upperLimit:
                        torque = joint.GetMotorTorque(60)
                        moment_of_inertia = body_to_check.inertia
                        angular_velocity = body_to_check.angularVelocity
                        angular_inertia = moment_of_inertia * angular_velocity

                        world_center = body_to_check.worldCenter
                        anchor = joint.anchorB
                        lever_vector = world_center - anchor
                        force_at_center = Box2D.b2Cross(lever_vector, -torque)

                        push_dot = Box2D.b2Dot(normal, force_at_center)
                        if push_dot > 0:
                            vel = torque + angular_inertia
                            push_mag = push_dot * self.push_mod * edge_length * density * vel * vel
                            push_force = np.clip(push_mag * -force_at_center, -self.max_push, self.max_push)
                            body_to_check.ApplyForce(force=push_force, point=anchor, wake=True)


class WaterContactDetector(contactListener):
    '''Tracks fixtures that are in contact with water.'''
    def __init__(self):
        super(WaterContactDetector, self).__init__()
        self.fixture_pairs = []

    def BeginContact(self, contact):
        fA, fB = contact.fixtureA, contact.fixtureB
        if fA.body.userData.object_type == CustomUserDataObjectTypes.WATER and \
           fB.body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT:
            self.fixture_pairs.append((fA, fB))
        elif fB.body.userData.object_type == CustomUserDataObjectTypes.WATER and \
             fA.body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT:
            self.fixture_pairs.append((fB, fA))

    def EndContact(self, contact):
        '''Safely remove fixture pairs when contact ends.'''
        fA, fB = contact.fixtureA, contact.fixtureB

        # START FIX: Thêm các bước kiểm tra an toàn toàn diện để ngăn lỗi sập
        if not (hasattr(fA, 'body') and hasattr(fB, 'body') and fA.body and fB.body and
                hasattr(fA.body, 'userData') and hasattr(fB.body, 'userData') and
                fA.body.userData and fB.body.userData):
            return
        # END FIX

        pair_to_remove = None
        if fA.body.userData.object_type == CustomUserDataObjectTypes.WATER and \
           fB.body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT:
            pair_to_remove = (fA, fB)
        elif fB.body.userData.object_type == CustomUserDataObjectTypes.WATER and \
             fA.body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT:
            pair_to_remove = (fB, fA)

        if pair_to_remove and pair_to_remove in self.fixture_pairs:
            self.fixture_pairs.remove(pair_to_remove)

    def Reset(self):
        '''Clear all stored contacts.'''
        self.fixture_pairs = []