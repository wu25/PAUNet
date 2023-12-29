# #!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Importing time for time management
import time

import numpy as np
import torch.nn as nn


class PID(nn.Module):
# class PID(nn.Module):


    def __init__(self, kp, ki, kd, direction):
        super(PID, self).__init__()
        self.direction = direction
        self.kp=kp
        self.ki=ki
        self.kd=kd
        self.updateTime = 1000
        self.lastUpdate = int(round(time.time() * 1000))
        self.output = 0.0
        self.pOutput = 0.0
        self.iOutput = 0.0
        self.dOutput = 0.0
        self.lastActual = 0.0
        # self.lowerIntegralLimit = 0
        # self.upperIntegralLimit = 0
        self.lastError=0.0
        if self.direction < 0:
            self.kp = kp * -1
            self.ki = ki * -1
            self.kd = kd * -1
        else:
            self.kp = kp
            self.ki = ki
            self.kd = kd
        self.lowerOuputLimit = np.zeros((4, 48, 180 * 180))
        self.upperOutputLimit = np.ones((4, 48, 180 * 180))
    def forward(self, target, actual):

        """Calulates the output based on the PID algorithm

        Parameters
        ----------
        target : float
            Desired value
        actual : float
            Current value

        Returns
        -------
        output : float
            The output correction
        """

        now = int(round(time.time() * 1000))
        timeDifference = now - self.lastUpdate
        print("timeDifference:",timeDifference)
        if timeDifference >= self.updateTime:
            error = target - actual
            print("内的error:",error)

            self.pOutput = error * self.kp
            self.iOutput += error * self.ki
            # self.dOutput += actual - self.lastActual
            self.dOutput=self.kd*(self.lastError-error)
            # if self.iOutput.any() < self.lowerIntegralLimit:
            #     self.iOutput = self.lowerIntegralLimit
            # elif self.iOutput.any() > self.upperIntegralLimit:
            #     self.iOutput = self.upperIntegralLimit

            # self.output = self.outputOffset + self.pOutput + self.iOutput + self.dOutput
            self.output =  self.pOutput + self.iOutput + self.dOutput
            print("pOutput",self.pOutput)
            print("iOutput",self.iOutput)
            print("dOutput",self.dOutput)
            print("nei de output:",self.output)
            # if self.output.any() < self.lowerOuputLimit.any():
            #     self.output = self.lowerOuputLimit
            # elif self.output.any() > self.upperOutputLimit.any():
            #     self.output = self.upperOutputLimit

            self.lastActual = actual
            self.lastError=error
            self.lastUpdate = now
            return self.output
        else:
            return self.output


# Example of usage
if __name__ == "__main__":
    pid = PID(kp=1.3, ki=1.5, kd=1.1, direction=1)
    pid.addOffset(50.0)
    pid.updateTime = 100
    pid.setOutputLimits(0, 100)

    while(True):
        output = pid.compute(50, 23)
        print(output)
