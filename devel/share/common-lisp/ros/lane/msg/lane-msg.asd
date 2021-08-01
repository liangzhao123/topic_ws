
(cl:in-package :asdf)

(defsystem "lane-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "lane" :depends-on ("_package_lane"))
    (:file "_package_lane" :depends-on ("_package"))
    (:file "lane" :depends-on ("_package_lane"))
    (:file "_package_lane" :depends-on ("_package"))
  ))