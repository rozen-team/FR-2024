import pigpio

class GPIOHandler:
    TYPE_A_PIN = 22
    TYPE_B_PIN = 27

    def __init__(self):
        self.pi = pigpio.pi()
        self.pi.set_mode(self.TYPE_A_PIN, pigpio.OUTPUT)
        self.pi.set_mode(self.TYPE_B_PIN, pigpio.OUTPUT)

        self.clear()

    def clear(self):
        self.pi.write(self.TYPE_A_PIN, 0)
        self.pi.write(self.TYPE_B_PIN, 0)

    def push_object(self, type_o):
        a_pin = (1 if type_o == 'A' else 0)
        b_pin = (not a_pin) 

        self.pi.write(self.TYPE_A_PIN, a_pin)
        self.pi.write(self.TYPE_B_PIN, b_pin)
        time.sleep(0.005)
        self.clear()