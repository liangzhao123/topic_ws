; Auto-generated. Do not edit!


(cl:in-package lane-msg)


;//! \htmlinclude lane.msg.html

(cl:defclass <lane> (roslisp-msg-protocol:ros-message)
  ((angle
    :reader angle
    :initarg :angle
    :type cl:fixnum
    :initform 0)
   (delta_x
    :reader delta_x
    :initarg :delta_x
    :type cl:fixnum
    :initform 0))
)

(cl:defclass lane (<lane>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <lane>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'lane)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name lane-msg:<lane> is deprecated: use lane-msg:lane instead.")))

(cl:ensure-generic-function 'angle-val :lambda-list '(m))
(cl:defmethod angle-val ((m <lane>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader lane-msg:angle-val is deprecated.  Use lane-msg:angle instead.")
  (angle m))

(cl:ensure-generic-function 'delta_x-val :lambda-list '(m))
(cl:defmethod delta_x-val ((m <lane>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader lane-msg:delta_x-val is deprecated.  Use lane-msg:delta_x instead.")
  (delta_x m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <lane>) ostream)
  "Serializes a message object of type '<lane>"
  (cl:let* ((signed (cl:slot-value msg 'angle)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'delta_x)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <lane>) istream)
  "Deserializes a message object of type '<lane>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'angle) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'delta_x) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<lane>)))
  "Returns string type for a message object of type '<lane>"
  "lane/lane")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'lane)))
  "Returns string type for a message object of type 'lane"
  "lane/lane")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<lane>)))
  "Returns md5sum for a message object of type '<lane>"
  "c27dae8bfb83971b171f1e3e63f6b5f6")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'lane)))
  "Returns md5sum for a message object of type 'lane"
  "c27dae8bfb83971b171f1e3e63f6b5f6")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<lane>)))
  "Returns full string definition for message of type '<lane>"
  (cl:format cl:nil "int8  angle~%int8  delta_x~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'lane)))
  "Returns full string definition for message of type 'lane"
  (cl:format cl:nil "int8  angle~%int8  delta_x~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <lane>))
  (cl:+ 0
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <lane>))
  "Converts a ROS message object to a list"
  (cl:list 'lane
    (cl:cons ':angle (angle msg))
    (cl:cons ':delta_x (delta_x msg))
))
