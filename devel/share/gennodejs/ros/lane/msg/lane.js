// Auto-generated. Do not edit!

// (in-package lane.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class lane {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.angle = null;
      this.delta_x = null;
    }
    else {
      if (initObj.hasOwnProperty('angle')) {
        this.angle = initObj.angle
      }
      else {
        this.angle = 0;
      }
      if (initObj.hasOwnProperty('delta_x')) {
        this.delta_x = initObj.delta_x
      }
      else {
        this.delta_x = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type lane
    // Serialize message field [angle]
    bufferOffset = _serializer.int8(obj.angle, buffer, bufferOffset);
    // Serialize message field [delta_x]
    bufferOffset = _serializer.int8(obj.delta_x, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type lane
    let len;
    let data = new lane(null);
    // Deserialize message field [angle]
    data.angle = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [delta_x]
    data.delta_x = _deserializer.int8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 2;
  }

  static datatype() {
    // Returns string type for a message object
    return 'lane/lane';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'c27dae8bfb83971b171f1e3e63f6b5f6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int8  angle
    int8  delta_x
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new lane(null);
    if (msg.angle !== undefined) {
      resolved.angle = msg.angle;
    }
    else {
      resolved.angle = 0
    }

    if (msg.delta_x !== undefined) {
      resolved.delta_x = msg.delta_x;
    }
    else {
      resolved.delta_x = 0
    }

    return resolved;
    }
};

module.exports = lane;
