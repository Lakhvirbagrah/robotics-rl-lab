#!/usr/bin/env python
import rospy 
from std_msgs.msg import String
mydata=''  
    
def callback(data):
    
    global mydata
    mydata=str(data.data)
    #print(mydata) 
    #print(type(mydata))

def listener():
    rospy.init_node('listener')
    rospy.Subscriber('sys_pub',String,callback)
    rate=rospy.Rate(1)
    while not rospy.is_shutdown():

        print(mydata)
        rate.sleep()
    rospy.spin()

listener()
