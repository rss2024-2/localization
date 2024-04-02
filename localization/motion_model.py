import numpy as np

class MotionModel:

    def __init__(self,node):
        self.covariance=np.eye(3)/10

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

        noise=np.random.multivariate_normal(np.zeros(3),self.covariance,N)

        dx=odometry[0]
        dy=odometry[1]
        dtheta=odometry[2]

        theta=particles[:,2]

        rot_x=dx*np.cos(theta)-dy*np.sin(theta)
        rot_y=dx*np.sin(theta)+dy*np.cos(theta)

        return particles+np.column_stack((rot_x,rot_y,np.full((N),dtheta)))+noise

# Test Case
# model=MotionModel()
# print(model.evaluate(np.array([[3,4,np.pi/6],[3,4,np.pi/3]]),np.array([0.223,-0.013,np.pi/60])))