import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/root/Capstone/AMD-Capstone/src/install/four_wheel_sim'
