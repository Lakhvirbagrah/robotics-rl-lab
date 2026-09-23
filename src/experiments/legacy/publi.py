import rospy 
from std_msgs.msg import String

def publisher():
    rospy.init_node('publi')
    pub=rospy.Publisher('publisher',String,queue_size=1)
    counter=0
    rate=rospy.Rate(1)
    while not rospy.is_shutdown():
        msg=str(counter)
        counter+=1
        pub.publish(msg)
        rate.sleep()



if __name__=='__main__':
    publisher()