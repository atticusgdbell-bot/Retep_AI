# Northstar alert thresholds

THERMAL_WARNING_C = 65
THERMAL_SHUTDOWN_C = 70
BATTERY_LOW_PERCENT = 20
SATELLITE_RETRY_SECONDS = 90

def should_shutdown(temperature_c):
    return temperature_c >= THERMAL_SHUTDOWN_C

def battery_is_low(percent):
    return percent <= BATTERY_LOW_PERCENT
