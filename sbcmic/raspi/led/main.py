"""
main.py - main runnable script for using the ReSpeaker 4-Mic array on the
Raspberry Pi.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import time

from apa102 import APA102
from gpiozero import LED


def main():
    leds = LED(5)
    leds.on()
    
    dev = APA102(num_led=12, global_brightness=10)

    for i in range(12):
        dev.clear_strip()
        dev.set_pixel(i, 255, 255, 255)
        dev.show()
        time.sleep(1)


if __name__ == '__main__':
    main()
