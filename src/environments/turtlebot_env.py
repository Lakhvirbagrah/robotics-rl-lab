import rospy
from perception.yolo_state import YoloStateProvider
from robots.turtlebot3 import TurtleBot3


class TurtlebotEnv:
    def __init__(self, task, action_duration=0.2):
        self.task = task
        self.action_duration = action_duration
        self.observation = None
        self.perception = YoloStateProvider()

        

        self.robot = TurtleBot3()

        rospy.sleep(1)

    

    def execute_action(self, action_index):
        action = self.task.get_action(action_index)

        self.robot.execute_action(action)

    def step(self, action):
        
        self.execute_action(action)

        rospy.sleep(self.action_duration)
        observation = self.perception.get_observation()

        if observation is None:
            return None, -1.0, False

        

        next_state = self.task.get_state(observation)
        reward = self.task.compute_reward(observation)
        done = self.task.is_done(observation)

        return next_state, reward, done

    def reset(self):
        self.robot.stop()
        self.task.reset()

        rospy.sleep(0.2)

        observation = self.perception.get_observation()

        while observation is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            observation = self.perception.get_observation()

        return self.task.get_state(observation)