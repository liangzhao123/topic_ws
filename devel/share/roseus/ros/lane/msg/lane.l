;; Auto-generated. Do not edit!


(when (boundp 'lane::lane)
  (if (not (find-package "LANE"))
    (make-package "LANE"))
  (shadow 'lane (find-package "LANE")))
(unless (find-package "LANE::LANE")
  (make-package "LANE::LANE"))

(in-package "ROS")
;;//! \htmlinclude lane.msg.html


(defclass lane::lane
  :super ros::object
  :slots (_angle _delta_x ))

(defmethod lane::lane
  (:init
   (&key
    ((:angle __angle) 0)
    ((:delta_x __delta_x) 0)
    )
   (send-super :init)
   (setq _angle (round __angle))
   (setq _delta_x (round __delta_x))
   self)
  (:angle
   (&optional __angle)
   (if __angle (setq _angle __angle)) _angle)
  (:delta_x
   (&optional __delta_x)
   (if __delta_x (setq _delta_x __delta_x)) _delta_x)
  (:serialization-length
   ()
   (+
    ;; int8 _angle
    1
    ;; int8 _delta_x
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int8 _angle
       (write-byte _angle s)
     ;; int8 _delta_x
       (write-byte _delta_x s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int8 _angle
     (setq _angle (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _angle 127) (setq _angle (- _angle 256)))
   ;; int8 _delta_x
     (setq _delta_x (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _delta_x 127) (setq _delta_x (- _delta_x 256)))
   ;;
   self)
  )

(setf (get lane::lane :md5sum-) "c27dae8bfb83971b171f1e3e63f6b5f6")
(setf (get lane::lane :datatype-) "lane/lane")
(setf (get lane::lane :definition-)
      "int8  angle
int8  delta_x

")



(provide :lane/lane "c27dae8bfb83971b171f1e3e63f6b5f6")


