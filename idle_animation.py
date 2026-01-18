from pythonosc.udp_client import SimpleUDPClient
import time
import math
import threading

client = SimpleUDPClient("127.0.0.1", 39539)


animation_running = False
animation_thread = None

def set_relaxed_pose():

    send_bone("LeftUpperArm", -0.08616048097610474, -0.0122603178024292, 7.450580596923828e-09, 0.012752960436046124, -0.013275292702019215, -0.6381993293762207, -0.7696511149406433)
    send_bone("RightUpperArm", 0.08616048097610474, -0.0122603178024292, 7.450580596923828e-09, 0.012754018418490887, 0.013277653604745865, 0.6381993889808655, -0.7696509957313538)
    
    send_bone("LeftLowerArm",-0.2198527455329895, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    send_bone("RightLowerArm", 0.2198527455329895, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    send_bone('LeftHand', -0.21468493342399597, 2.384185791015625e-06, 0.0003759637475013733, 0.0, 0.0, 0.0, 1.0)
    send_bone('RightHand', 0.21468493342399597, 2.384185791015625e-06, 0.0003759637475013733, 0.0, 0.0, 0.0, 1.0)
    #client.send_message("/VMC/Ext/Blend/Apply", [])

def send_bone(bone_name, px, py, pz, qx, qy, qz, qw):
    """Send bone position and rotation to VSeeFace"""
    client.send_message("/VMC/Ext/Bone/Pos", [
        bone_name,
        px, py, pz,  # position
        qx, qy, qz, qw  # rotation (quaternion)
    ])

def idle_animation():
    """Simple breathing and head movement idle animation"""
    global animation_running
    
    # Animation parameters
    time_elapsed = 0
    breath_speed = 10.0 # Breathing cycle speed
    head_sway_speed = 0.8  # Head movement speed
    
    while animation_running:
        head_sway_x = math.sin(time_elapsed * head_sway_speed) * 0.055
        head_sway_y = math.cos(time_elapsed * head_sway_speed * 0.7) * 0.030
        
        # Breathing
        breath = math.sin(time_elapsed * breath_speed) * 0.04
        arm_breath = breath * 7

        # Sway
        sway = math.sin(time_elapsed * 0.8) * 0.03
        twist = math.sin(time_elapsed * 0.6) * 0.08


        
        send_bone("Head", 
                 head_sway_x,
                 breath,
                 0.0,
                 head_sway_y,
                 head_sway_x * 0.5,   
                 0.0,               
                 1.0)                    
        
        send_bone("Pelvis",
                sway * 0.2,
                0.0,
                0.0,
                0.0, twist * 0.1, 0.0, 1.0)

        send_bone("Spine",
                sway * 0.4,
                breath,
                0.0,
                0.0, twist * 0.2, 0.0, 1.0)

        send_bone("UpperChest",
                sway * 0.6,
                breath * 1.2,
                0.0,
                0.0, twist * 0.3, 0.0, 1.0)
        send_bone("LeftShoulder",  0, arm_breath * 0.3, 0,  0.0, 0.0,  0.08, 0.996)
        send_bone("RightShoulder", 0, arm_breath * 0.3, 0,  0.0, 0.0, -0.08, 0.996)
        send_bone("LeftUpperArm",  0, arm_breath, 0,  0.65, 0.0, -0.20, 0.73)
        send_bone("RightUpperArm", 0, arm_breath, 0,  0.65, 0.0,  0.20, 0.73)
        
        set_relaxed_pose()

        
        time_elapsed += 0.06
        
        time.sleep(0.03)
    

    send_bone("Head", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    send_bone("Pelvis", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    send_bone("Spine", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    send_bone("UpperChest", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    send_bone('LeftShoulder', -0.022395402193069458, 0.10568630695343018, -0.029968924820423126, -0.0, -0.0, -1.2906511009065712e-16, 1.0)
    send_bone('RightShoulder', 0.022395402193069458, 0.10568630695343018, -0.029968924820423126, 1.598720986053636e-14, -0.0, -1.8626448650138627e-07, 1.0)
    send_bone('LeftUpperArm', -0.08616048097610474, -0.0122603178024292, 7.450580596923828e-09, -0.00035142849083058536, -0.002589394571259618, 0.0022767814807593822, 0.9999940395355225)
    send_bone('RightUpperArm', 0.08616048097610474, -0.0122603178024292, 7.450580596923828e-09, -0.0003602279757615179, 0.0025905065704137087, -0.0022766415495425463, 0.9999940395355225)
    send_bone('LeftLowerArm', -0.2198527455329895, 0.0, 0.0, -0.005260533187538385, -0.0002524475276004523, 1.012035136227496e-05, 0.9999861717224121)
    send_bone('RightLowerArm', 0.2198527455329895, 0.0, 0.0, -0.0052609494887292385, 0.00025121073122136295, -1.0190046850766521e-05, 0.9999861717224121)
    send_bone('LeftHand', -0.21468493342399597, 2.384185791015625e-06, 0.0003759637475013733, -0.003351254388689995, -2.3285681436391314e-06, 0.00024986971402540803, 0.9999943971633911)
    send_bone('RightHand', 0.21468493342399597, 2.384185791015625e-06, 0.0003759637475013733, -0.003341992385685444, 2.612446905914112e-06, -0.00024966715136542916, 0.9999943971633911)

def start_idle_animation():
    """Start the idle animation in a background thread"""
    global animation_running, animation_thread
    
    if animation_running:
        #print("Animation already running!")
        return
    
    animation_running = True
    animation_thread = threading.Thread(target=idle_animation, daemon=True)
    animation_thread.start()
    #print("Started idle animation thread")

def stop_idle_animation():
    """Stop the idle animation"""
    global animation_running
    
    if not animation_running:
        #print("Animation not running!")
        return
    
    animation_running = False
    if animation_thread:
        animation_thread.join(timeout=1.0)

def set_expression_fast(expression, target_value, steps=5):
    """Quickly transition to target value"""
    for i in range(steps + 1):
        value = (i / steps) * target_value
        client.send_message("/VMC/Ext/Blend/Val", [expression, value])
        time.sleep(0.02)

def unset_expression_fast(expression, steps=5):
    set_expression_fast(expression, 0.0, steps)

if __name__ == "__main__":
    start_idle_animation()
    
    try:
        print("\nIdle animation continues... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping everything...")
        stop_idle_animation()
        print("Goodbye!")