"""Measured-state motion scheduler. Does not modify physics, weights or motions."""
import numpy as np
from scipy.spatial.transform import Rotation as R


def yaw(q):
    return float(R.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_euler('xyz')[2])


def wrap(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))


def ease(x):
    x = np.clip(x, 0., 1.)
    return x**3 * (10 - 15*x + 6*x*x)


class StateAwareScheduler:
    def __init__(self, bank, point_ids, actual_ids, offset, floor):
        self.bank, self.pi, self.ai = bank, point_ids, actual_ids
        self.ref = bank['straight_lr']
        self.origin = self.ref['body_pos_w'][0, 0].astype(float)
        self.guard_p = self.ref['body_pos_w'][0, point_ids].astype(float)
        self.guard_q = self.ref['body_quat_w'][0, point_ids].astype(float)
        self.offset, self.floor = offset, floor
        self.height = self.origin[2] + floor
        self.active, self.phase = None, 0
        self.last_p = self.last_q = None
        self.events, self.stable_ticks, self.tick = [], 0, 0
        self.mode = 'normal'
        self.recovery_count = 0
        self.interruptions = 0
        self.tracking_warning_ticks = 0
        self.rate = 1.
        self.settings = dict(forward_rate=1.7,turn_rate=1.8,attack_bearing_deg=22.,attack_feet_deg=32.,
                             straight_max_m=.60,uppercut_max_m=.45,approach_target_m=.54,
                             recovery_min_s=.20,recovery_stable_s=.10,
                             straight_nominal_rate=1.,uppercut_rate=39/25,
                             straight_original_frames=[0,12,22,32,34,46,56,66],
                             straight_control_frames=[0,8,14,21,23,31,37,44])
        self.settings.update(alternating_interval_s=.25,arm_period_s=.50,
                             burst_source_frames=[0,12,12,0,0],burst_control_frames=[0,4,9,14,25])
        self.next_review=25
        self.anchor_transition=None
        self.nominal_last_p=None
        self.wrist_vz=np.zeros(2)
        self.settings.update(wrist_tracking_gain=0.,wrist_tracking_limit_m=.12,
                             uppercut_wrist_tracking_gain=0.)
        self.settings['straight_forward_compensation_m']=.40
        self.settings.update(vertical_gain=2.,vertical_damping_s=.15,vertical_limit_m=.25,
                             tracking_soft_limit_m=.19,tracking_hard_limit_m=.24,
                             tracking_warning_frames=3,stable_pelvis_error_m=.17,
                             stable_foot_error_m=.18,stable_speed_m_s=.85,
                             stable_foot_speed_m_s=1.8,
                             travel_tracking_soft_limit_m=.18,travel_stable_pelvis_error_m=.16,
                             travel_stable_foot_error_m=.17,travel_stable_speed_m_s=.8,
                             travel_stable_foot_speed_m_s=1.6,
                             settle_relaxation_scope='straight_lr only')
        self.settings.update(straight_anchor='measured pelvis and support feet',
                             recovery_refresh='physically settled; every 0.2s')

    def capture_straight_support(self,p,q):
        self.straight_feet_p=p[3:].copy()
        self.straight_feet_p[:,2]=self.guard_p[3:,2]+self.floor
        self.straight_feet_q=q[3:].copy()
        for j in [3,4]:
            rot=R.from_euler('z',wrap(yaw(q[j])-yaw(self.guard_q[j])))
            self.straight_feet_q[j-3]=(rot*R.from_quat(self.guard_q[j,[1,2,3,0]])).as_quat()[[3,0,1,2]]

    def refresh_recovery(self,p,q):
        self.anchor(p,q)
        self.position[:2]=p[0,:2]
        gp,gq=self.transform(self.ref,np.array([0]))
        self.recover_p,self.recover_q=gp[0],gq[0]
        self.recover_p[3:,:2]=p[3:,:2]
        for j in [3,4]:
            rot=R.from_euler('z',wrap(yaw(q[j])-yaw(self.guard_q[j])))
            self.recover_q[j]=(rot*R.from_quat(self.guard_q[j,[1,2,3,0]])).as_quat()[[3,0,1,2]]

    def wrist_correction(self,actual):
        correction=np.zeros((2,3))
        if self.active=='straight_lr' and self.nominal_last_p is not None:
            # Actual wrist height and vertical speed suppress the repeated
            # down-then-up oscillation without changing the forward target.
            error=self.nominal_last_p[1:3,2]-actual[1:3,2]
            limit=self.settings['vertical_limit_m']
            correction[:,2]=np.clip(self.settings['vertical_gain']*error-
                self.settings['vertical_damping_s']*self.wrist_vz,-limit,limit)
        return correction

    def hand_time(self, times, hand):
        period=self.settings['arm_period_s']/.02
        t=np.asarray(times,dtype=float)-period*.5*hand
        source=np.interp(np.mod(t,period),self.settings['burst_control_frames'],self.settings['burst_source_frames'])
        return np.where(t<0,0,source)+34*hand

    def attack_targets(self, times):
        if self.active=='straight_lr':
            z=self.bank[self.active]
            p,q=self.transform(z,np.zeros(len(times)))
            for hand in [0,1]:
                pp,qq=self.transform(z,self.hand_time(times,hand))
                p[:,hand+1]=pp[:,hand+1];q[:,hand+1]=qq[:,hand+1]
            guard,_=self.transform(z,np.zeros(len(times)))
            heading=yaw(self.guard_q[0])+self.offset
            forward=self.rotation.apply(np.array([np.cos(heading),np.sin(heading),0.]))
            # Calibrated command pre-emphasis, not a physically reachable pose:
            # retain the accepted height/orientation/lateral path and drive only
            # along the chest-facing axis. Fade to zero on return to guard.
            for hand in [0,1]:
                extension=np.clip(np.linalg.norm(p[:,hand+1]-guard[:,hand+1],axis=-1)/.28,0.,1.)
                p[:,hand+1]+=self.settings['straight_forward_compensation_m']*extension[:,None]*forward
            p[:,3:]=self.straight_feet_p;q[:,3:]=self.straight_feet_q
            return p,q
        return self.transform(self.bank[self.active],np.mod(times,25)*(39/25))

    def continue_attack(self,p,q,info):
        old_p=self.position.copy();old_yaw=self.rotation.as_euler('xyz')[2]
        self.anchor(p,q,info.get('heading_lead_deg',0.))
        if self.active=='straight_lr':
            self.position[:2]=p[0,:2]
            if info['foot_speed_m_s']<.2 and info['tilt_deg']<11:
                self.capture_straight_support(p,q)
        delta=self.position-old_p;delta*=min(1.,.04/max(np.linalg.norm(delta),1e-9))
        new_yaw=old_yaw+np.clip(wrap(self.rotation.as_euler('xyz')[2]-old_yaw),-np.radians(4),np.radians(4))
        self.anchor_transition=(old_p,old_p+delta,old_yaw,new_yaw,self.phase)
        self.position=old_p;self.rotation=R.from_euler('z',old_yaw)
        self.next_review+=self.settings['arm_period_s']/.02 if self.active=='straight_lr' else 25
        self.events.append(dict(frame=self.tick,time_s=self.tick*.02,skill=self.active,reason='continuous_alternating_attack',continuation=True,self_state=info.copy()))

    def motion_time(self, skill, control_steps):
        if skill=='straight_lr':
            assert len(self.bank[skill]['joint_pos'])==67,'Straight timing requires the verified 67-frame motion'
            return np.interp(control_steps,self.settings['straight_control_frames'],self.settings['straight_original_frames'])
        return np.asarray(control_steps)*self.rate

    def duration_steps(self):
        if self.active in ['straight_lr','uppercut_lr']:return self.next_review
        return (len(self.bank[self.active]['joint_pos'])-1)/self.rate

    def observe(self, raw):
        p = raw['position'][0, 0, self.ai].cpu().numpy().copy()
        q = raw['quaternion'][0, 0, self.ai].cpu().numpy().copy()
        v = raw['velocity'][0, 0, self.ai].cpu().numpy().copy()
        self.wrist_vz=v[1:3,2].copy()
        opponent = raw['position'][0, 1, 0].cpu().numpy()
        tilt = np.degrees(np.arccos(np.clip(R.from_quat(q[0, [1,2,3,0]]).as_matrix()[2,2], -1, 1)))
        errors = np.zeros(5) if self.last_p is None else np.linalg.norm(p-self.last_p, axis=-1)
        delta = opponent[:2]-p[0,:2]
        heading = yaw(q[0])+self.offset
        foot_yaws = [yaw(q[j])+wrap(yaw(self.guard_q[0])+self.offset-yaw(self.guard_q[j])) for j in [3,4]]
        feet_heading = np.arctan2(np.sin(foot_yaws).sum(), np.cos(foot_yaws).sum())
        target_heading = np.arctan2(delta[1],delta[0])
        info = dict(tilt_deg=float(tilt), height_m=float(p[0,2]), speed_m_s=float(np.linalg.norm(v[0,:2])),
                    foot_speed_m_s=float(np.max(np.linalg.norm(v[3:,:2],axis=-1))),
                    pelvis_error_m=float(errors[0]), foot_error_m=float(max(errors[3:])),
                    distance_m=float(np.linalg.norm(delta)), bearing_deg=float(np.degrees(wrap(target_heading-heading))),
                    feet_error_deg=float(np.degrees(wrap(target_heading-feet_heading))))
        # Position error is checked before any reference correction; rebasing cannot hide a fault.
        tracking=max(errors[0],max(errors[3:]))
        prefix='' if self.active=='straight_lr' else 'travel_'
        warning=tracking>self.settings[prefix+'tracking_soft_limit_m']
        self.tracking_warning_ticks=self.tracking_warning_ticks+1 if warning else 0
        # Tilt, height loss and large position errors still interrupt immediately.
        hard=bool(tilt>17 or p[0,2]<self.height-.09 or tracking>self.settings['tracking_hard_limit_m'])
        info['tracking_warning_ticks']=self.tracking_warning_ticks
        info['hard_unsafe']=hard
        info['unsafe'] = bool(hard or self.tracking_warning_ticks>=self.settings['tracking_warning_frames'])
        info['stable'] = bool(not info['unsafe'] and tilt<13 and p[0,2]>self.height-.07
                              and info['speed_m_s']<self.settings[prefix+'stable_speed_m_s']
                              and info['foot_speed_m_s']<self.settings[prefix+'stable_foot_speed_m_s']
                              and errors[0]<self.settings[prefix+'stable_pelvis_error_m']
                              and max(errors[3:])<self.settings[prefix+'stable_foot_error_m'])
        info['heading_lead_deg'] = 0. if self.last_q is None else float(np.clip(np.degrees(wrap(yaw(self.last_q[0])-yaw(q[0]))),-12,12))
        return p, q, info

    def anchor(self, p, q, lead=0.):
        self.rotation = R.from_euler('z', wrap(yaw(q[0])-yaw(self.guard_q[0])+np.radians(lead)))
        # Current support centre, not the previous clip's ideal endpoint.
        feet_offset = self.rotation.apply(self.guard_p[3:].mean(0)-self.origin)
        self.position = p[0].astype(float).copy()
        self.position[:2] = p[3:,:2].mean(0)-feet_offset[:2]
        self.position[2] = self.height

    def transform(self, z, indices):
        # Fractional motion time preserves 50Hz output and the 20-100ms lookahead.
        t=np.clip(np.asarray(indices,dtype=float),0,len(z['joint_pos'])-1)
        lo=np.floor(t).astype(int);hi=np.minimum(lo+1,len(z['joint_pos'])-1);a=(t-lo)[:,None,None]
        p=(1-a)*z['body_pos_w'][lo][:,self.pi]+a*z['body_pos_w'][hi][:,self.pi]
        q0=z['body_quat_w'][lo][:,self.pi];q1=z['body_quat_w'][hi][:,self.pi]
        q1=np.where(np.sum(q0*q1,axis=-1,keepdims=True)<0,-q1,q1)
        q=(1-a)*q0+a*q1;q/=np.linalg.norm(q,axis=-1,keepdims=True)
        p=self.rotation.apply((p-self.origin).reshape(-1,3)).reshape(p.shape)+self.position
        q=(self.rotation*R.from_quat(q.reshape(-1,4)[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]].reshape(q.shape)
        return p,q

    def begin(self, skill, p, q, info, reason):
        self.anchor(p,q,0. if skill=='recover_guard' else info.get('heading_lead_deg',0.))
        self.active, self.phase = skill, 0
        if skill=='straight_lr':
            self.position[:2]=p[0,:2]
            self.capture_straight_support(p,q)
        self.next_review=self.settings['arm_period_s']/.02 if skill=='straight_lr' else 25
        self.anchor_transition=None
        self.nominal_last_p=None
        self.rate=self.settings['forward_rate'] if skill.startswith('forward') else self.settings['turn_rate'] if skill.startswith('turn') else 1.
        if skill=='straight_lr':self.rate=self.settings['straight_nominal_rate']
        if skill=='uppercut_lr':self.rate=self.settings['uppercut_rate']
        self.start_p = (p if self.last_p is None else self.last_p).copy()
        self.start_q = (q if self.last_q is None else self.last_q).copy()
        if skill=='recover_guard':
            self.mode='recover';self.recovery_count+=1;self.stable_ticks=0
            gp,gq=self.transform(self.ref,np.array([0]))
            self.recover_p,self.recover_q=gp[0],gq[0]
            self.recover_p[3:,:2]=p[3:,:2]
            for j in [3,4]:
                # Keep the current foot heading while requesting a flat sole.
                rot=R.from_euler('z',wrap(yaw(q[j])-yaw(self.guard_q[j])))
                self.recover_q[j]=(rot*R.from_quat(self.guard_q[j,[1,2,3,0]])).as_quat()[[3,0,1,2]]
        else:self.mode='normal'
        self.events.append(dict(frame=self.tick,time_s=self.tick*.02,skill=skill,reason=reason,playback_rate=self.rate,self_state=info.copy()))

    def choose(self, info):
        if info['unsafe']:return 'recover_guard','unstable'
        if not info['stable']:return 'recover_guard','not_settled'
        d,pe,fe=info['distance_m'],info['bearing_deg'],info['feet_error_deg']
        # Attack before fine alignment: a usable direction is sufficient.
        if d<=self.settings['straight_max_m'] and abs(pe)<=self.settings['attack_bearing_deg'] and abs(fe)<=self.settings['attack_feet_deg']:
            return ('straight_lr' if d>self.settings['uppercut_max_m'] else 'uppercut_lr'),'attack_in_cone'
        gate=22 if d<=self.settings['straight_max_m'] else 25
        if abs(pe)>gate or abs(fe)>gate+4:
            correction=.75*pe+.25*fe-info['heading_lead_deg']
            if abs(correction)<2.5:return 'settle_guard','heading_deadband'
            side='left' if correction>0 else 'right'
            angle=min([5,10,20],key=lambda x:abs(x-abs(correction)))
            return (f'turn_{side}' if angle==20 else f'turn_{side}_{angle:02}'),'align_from_actual_heading'
        if d>self.settings['straight_max_m']:
            budget=min(.20 if d>.95 else .12,max(.03,d-self.settings['approach_target_m']-info['speed_m_s']*.10))
            cm=max(x for x in [3,5,8,12,20] if x/100<=budget+1e-8)
            return ('forward_short' if cm==20 else f'forward_{cm:02}cm'),'approach_opponent_from_current_support'
        return ('straight_lr' if d>self.settings['uppercut_max_m'] else 'uppercut_lr'),'stable_attack'

    def step(self,raw,tick):
        self.tick=tick;p,q,info=self.observe(raw)
        self.stable_ticks=self.stable_ticks+1 if info['stable'] else 0
        if self.active is None:
            skill,reason=self.choose(info);self.begin(skill,p,q,info,reason)
        elif self.mode!='recover' and info['unsafe']:
            self.interruptions+=1;self.begin('recover_guard',p,q,info,'interrupt_unstable_motion')
        elif self.mode=='recover':
            # A stale target must not latch recovery after the body has settled.
            # Check physical posture/speed before updating the reference.
            if (self.phase>=10 and self.phase%10==0 and info['tilt_deg']<10
                    and info['height_m']>self.height-.07 and info['speed_m_s']<.35
                    and info['foot_speed_m_s']<.25):
                self.refresh_recovery(p,q)
            if self.phase>=10 and self.stable_ticks>=5:
                skill,reason=self.choose(info);self.begin(skill,p,q,info,'recovered_'+reason)
        elif self.phase>=self.duration_steps():
            skill,reason=self.choose(info)
            if skill==self.active and skill in ['straight_lr','uppercut_lr']:
                self.continue_attack(p,q,info)
            else:self.begin(skill,p,q,info,reason)
        if self.anchor_transition is not None:
            a,b,ya,yb,start=self.anchor_transition;u=ease((self.phase-start+1)/8)
            self.position=a*(1-u)+b*u;self.rotation=R.from_euler('z',ya+(yb-ya)*u)
            if u>=1:self.anchor_transition=None
        future=self.phase+np.arange(1,6)
        if self.mode=='recover':
            tp=np.repeat(self.recover_p[None],5,axis=0);tq=np.repeat(self.recover_q[None],5,axis=0)
        elif self.active in ['straight_lr','uppercut_lr']:
            tp,tq=self.attack_targets(future)
        else:
            z=self.bank[self.active];tp,tq=self.transform(z,self.motion_time(self.active,future))
        # Blend command changes rather than teleporting the body's target pose.
        blend=ease(future/(8 if self.mode=='recover' else 2 if self.active in ['straight_lr','uppercut_lr'] else 4 if self.rate>1 else 8))
        tp=self.start_p[None]*(1-blend[:,None,None])+tp*blend[:,None,None]
        dots=np.sum(tq*self.start_q[None],axis=-1,keepdims=True)
        tq=np.where(dots<0,-tq,tq)
        tq=self.start_q[None]*(1-blend[:,None,None])+tq*blend[:,None,None]
        tq/=np.linalg.norm(tq,axis=-1,keepdims=True)
        nominal=tp[0].copy()
        if self.active in ['straight_lr','uppercut_lr']:
            tp[:,1:3]+=self.wrist_correction(p)[None]
        self.nominal_last_p=nominal
        self.last_p,self.last_q=tp[0].copy(),tq[0].copy()
        self.phase+=1
        return tp,tq,self.active,info
