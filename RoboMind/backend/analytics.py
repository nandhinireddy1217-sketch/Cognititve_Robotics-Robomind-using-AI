import random
from datetime import datetime


def generate_sensor_data():

    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "temperature": round(random.uniform(24, 35), 2),
        "distance": round(random.uniform(10, 100), 2),
        "motion": random.choice([0, 1]),
        "battery": random.randint(70, 100)
    }