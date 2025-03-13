import can

# Actual message IDs found for now "steering angle for Advanced parking assist (APA) 00xBB, Throttle vlaye from Propulsion and braking commanfs in stock APA 0x24B" -- need to confirm with Zarah and team 

# Defining CAN message IDs for throttle and steering (using arbitrary values)
THROTTLE_CAN_ID = 0x101
STEERING_CAN_ID = 0x102

def convert_throttle_to_CAN(throttle: float) -> can.Message:
    """
    Converts a throttle value (assumed to be between 0 and 1) to a CAN message.
    The throttle value is scaled to a value between 0 and 255.
    """
    # Clamp throttle value between 0 and 1
    throttle = max(0.0, min(10.0, throttle))
    # Convert to an integer value (0-255)
    throttle_int = int((throttle/10.0) * 255)
    # Create a CAN message (assuming 1-byte data for throttle)
    data = [throttle_int]
    message = can.Message(arbitration_id=THROTTLE_CAN_ID, data=data, is_extended_id=False)
    return message

def convert_steering_to_CAN(steering_angle: float) -> can.Message:
    """
    Converts a steering angle (in degrees, for example) to a CAN message.
    We assume the steering angle ranges from -30 to +30 degrees.
    It is scaled to an integer between 0 and 255 where 127 represents zero steering.
    """
    # Clamp steering angle to valid range (-30, 30)
    steering_angle = max(-30.0, min(30.0, steering_angle))
    # Scale steering angle to 0-255. Zero steering is 127.
    # Here -30 degrees -> 0, +30 degrees -> 255
    steering_int = int((steering_angle + 30) / 60 * 255)
    data = [steering_int]
    message = can.Message(arbitration_id=STEERING_CAN_ID, data=data, is_extended_id=False)
    return message

def print_CAN_message(message: can.Message) -> str:
    """
    Utility function to print a CAN message in a human-readable format.
    """
    return f"CAN ID: {hex(message.arbitration_id)}, Data: {message.data}"
