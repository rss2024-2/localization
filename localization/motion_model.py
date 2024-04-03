import numpy as np

class MotionModel:

    def __init__(self,node):
        self.covariance=np.diag([0.1,np.pi/3])

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        N=len(particles)

        noise=np.random.multivariate_normal(np.zeros(2),self.covariance,N)

        dt=odometry[2]
        v=odometry[0]+noise[:,0]
        dtheta=(odometry[1]+noise[:,1])*dt

        theta=particles[:,2]

        return particles+np.column_stack((v*np.cos(theta)*dt,v*np.sin(theta)*dt,dtheta))

# Test Case
# model=MotionModel()
# print(model.evaluate(np.array([[3,4,np.pi/6],[3,4,np.pi/3]]),np.array([0.223,-0.013,np.pi/60])))