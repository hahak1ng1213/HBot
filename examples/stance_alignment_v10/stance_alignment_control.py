"""Measured five-point stance alignment; no learned weights or extra targets."""
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def wrap(x):return math.atan2(math.sin(x),math.cos(x))
def yaw(q):return float(R.from_quat(np.asarray(q)[[1,2,3,0]]).as_euler('xyz')[2])
def mean_angle(angles):return math.atan2(sum(math.sin(x) for x in angles),sum(math.cos(x) for x in angles))

def geometry(position,quaternion,feet_ids,config):
    pelvis_delta=position[1,0,:2]-position[0,0,:2]
    feet_delta=position[1,feet_ids,:2].mean(0)-position[0,feet_ids,:2].mean(0)
    pelvis_target=math.atan2(pelvis_delta[1],pelvis_delta[0])
    feet_target=math.atan2(feet_delta[1],feet_delta[0])
    pelvis_heading=yaw(quaternion[0,0])+config['pelvis_heading_offset_rad']
    feet_heading=mean_angle([yaw(quaternion[0,i])+v for i,v in zip(feet_ids,config['foot_heading_offsets_rad'])])
    distance=float(np.linalg.norm(pelvis_delta))
    zone='attack' if distance<=config['near_distance_m'] else 'approach' if distance<=config['approach_alignment_distance_m'] else 'far'
    pe,fe=wrap(pelvis_target-pelvis_heading),wrap(feet_target-feet_heading)
    adjustment=np.clip(wrap(feet_target-pelvis_target),-math.radians(config['max_foot_target_adjustment_deg']),math.radians(config['max_foot_target_adjustment_deg']))
    desired=pelvis_target+config['foot_target_weight']*adjustment
    return dict(distance_m=distance,pelvis_error_deg=math.degrees(pe),feet_error_deg=math.degrees(fe),
                pelvis_target_rad=pelvis_target,feet_target_rad=feet_target,desired_heading_rad=float(desired),
                pelvis_heading_rad=pelvis_heading,feet_heading_rad=feet_heading,zone=zone,
                aligned=abs(math.degrees(pe))<=config[zone+'_pelvis_deg'] and abs(math.degrees(fe))<=config[zone+'_feet_deg'])

def choose(info,reference_heading,config,rules):
    if not info['aligned']:
        pe,fe=info['pelvis_error_deg'],info['feet_error_deg']
        zone=info['zone'];pg,fg=config[zone+'_pelvis_deg'],config[zone+'_feet_deg']
        desired=.75*pe+.25*fe
        low,high=max(pe-pg,fe-fg),min(pe+pg,fe+fg)
        if low<=high:desired=float(np.clip(desired,low,high))
        # Compensate the observed pelvis/reference yaw tracking offset. Merely
        # holding a geometrically aligned reference leaves a persistent actual
        # pelvis bias uncorrected. Bound compensation and command lead so a
        # transient lower-controller lag cannot accumulate unlimited rotation.
        lead=math.degrees(wrap(reference_heading-info['pelvis_heading_rad']))
        target_lead=float(np.clip(desired+np.clip(lead,-20.,20.),-25.,25.))
        correction=target_lead-lead
        if abs(correction)<=config['settling_reference_deadband_deg']:return 1,'settle_guard',correction
        angle=min(config['angle_choices_deg'],key=lambda v:abs(v-abs(correction)))
        side='left' if correction>0 else 'right'
        return 1,('turn_'+side if angle==20 else f'turn_{side}_{angle:02}'),correction
    distance=info['distance_m']
    if distance>rules['straight_max_m']:return 0,'forward_short',0.
    if distance>rules['uppercut_max_m']:return 2,'straight_lr',0.
    return 3,'uppercut_lr',0.
