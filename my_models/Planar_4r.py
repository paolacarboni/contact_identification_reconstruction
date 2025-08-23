import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteMDH, RevoluteDH
import numpy as np
from math import pi
import matplotlib.pyplot as plt

class Planar_4r:
    def __init__(self, link_lengths, link_masses):
        self.l1 = link_lengths[0]
        self.l2 = link_lengths[1]
        self.l3 = link_lengths[2]
        self.l4 = link_lengths[3]
        self.m1 = link_masses[0]
        self.m2 = link_masses[1]
        self.m3 = link_masses[2]
        self.m4 = link_masses[3]
        self.robot = self._create_robot()

    def _create_robot(self):
        #Define the links using the modified DH convention
        L1 = RevoluteMDH(a=0, alpha=0, d=0)
        L2 = RevoluteMDH(a=self.l1, alpha=0, d=0, offset=0)
        L3 = RevoluteMDH(a=self.l2, alpha=0, d=0)
        L4 = RevoluteMDH(a=self.l3, alpha=0, d=0)
        #L5 = RevoluteMDH(a=self.l4, alpha=0, d=0)


        #Construct the robot model
        robot = DHRobot([L1, L2, L3, L4], name='Planar_4r')
        return robot
    

    def plot(self, q=None, block=True):
        """
        Plot the robot model in 3D with optional joint configuration.

        :param q: List of joint angles in radians. If None, uses zero configuration.
        :param block: Whether to block execution until plot window is closed.
        """
        if q is None:
            q = [0, 0, 0, 0]
        self.robot.plot(q, block=block)