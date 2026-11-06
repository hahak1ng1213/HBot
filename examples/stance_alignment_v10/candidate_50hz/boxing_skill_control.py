"""Shared rule labels and runtime constraints for the four-skill selector."""
import math

DEFAULT_RULES = dict(
    turn_threshold_deg=30., straight_max_m=.60, uppercut_max_m=.50,
    too_close_m=.35, far_zone_m=1.5,
    priority=['turn if abs(bearing)>30deg', 'advance if distance>0.60m',
              'straight if distance>0.50m', 'uppercut otherwise'],
    too_close='still uppercut when aligned; independent too_close flag',
    boundary='30deg aligned; 0.60m straight; 0.50m uppercut',
    scheduling='Select only at completed skill boundaries; every-frame suggested labels are also saved',
    version='standard-five-range060-20261106',
    calibration_status='User requested +0.10m attack thresholds',
)

def validate_rules(rules):
    for key in ['turn_threshold_deg','straight_max_m','uppercut_max_m','too_close_m','far_zone_m']:
        if not math.isfinite(rules[key]):raise ValueError(key)
    if not 0 < rules['too_close_m'] <= rules['uppercut_max_m'] < rules['straight_max_m'] < rules['far_zone_m']:
        raise ValueError('Invalid distance ordering')
    if not 0 < rules['turn_threshold_deg'] < 180:raise ValueError('Invalid turn threshold')

def rule_class(distance_m, bearing_rad, rules=DEFAULT_RULES):
    """Class IDs: advance=0, turn=1, straight=2, uppercut=3."""
    if not math.isfinite(distance_m) or distance_m < 0 or not math.isfinite(bearing_rad):
        raise ValueError('Invalid opponent geometry')
    bearing_rad=math.atan2(math.sin(bearing_rad),math.cos(bearing_rad))
    if abs(bearing_rad)>math.radians(rules['turn_threshold_deg'])+1e-12:return 1
    if distance_m>rules['straight_max_m']:return 0
    return 2 if distance_m>rules['uppercut_max_m'] else 3

def enforce_selection(proposed_class, distance_m, bearing_rad, rules=DEFAULT_RULES):
    """Project model suggestion onto the user-defined angle/range class.

    The current rules allow exactly one class per state, so this constrained
    execution is rule-controlled. Raw model output is retained in decision logs.
    """
    if proposed_class not in range(4):raise ValueError('Invalid proposed class')
    result=rule_class(distance_m,bearing_rad,rules)
    return result, result != proposed_class
