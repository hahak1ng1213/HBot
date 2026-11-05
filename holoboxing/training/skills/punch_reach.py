"""Red-hand reach measurements against actual opponent torso/chest surfaces.

Read the current MuJoCo FK snapshot only. Call after ``physics.state()`` and
before automatic episode resets; this reader never advances the simulation.
Collision aliases and mesh semantic overrides are shared with the referee.
"""

import mujoco
import numpy as np
import torch

from holoboxing.evaluation.runtime.contact_geometry import (
    opponent_target_semantic,
    strike_geometry,
)

_VALUES = (
    "gap_m",
    "closing_speed_mps",
    "arm_aim_speed_mps",
    "arm_outward_speed_mps",
)
_FK_FIELDS = (
    "qpos",
    "qvel",
    "geom_xpos",
    "geom_xmat",
    "xpos",
    "xmat",
    "cvel",
    "subtree_com",
)


def _bindings(model):
    torso_ids = []
    for prefix in ("p1_", "p2_"):
        torso = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "torso_link"
        )
        if torso < 0:
            raise ValueError(f"Punch reach requires {prefix}torso_link")
        torso_ids.append(torso)
    hands, targets = [[], []], []
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        strike = strike_geometry(model, geom)
        if strike in ((0, "left_hand"), (0, "right_hand")):
            hands[int(strike[1] == "right_hand")].append(geom)
        target = opponent_target_semantic(model, 0, geom)
        if target is not None and target[2] in ("torso", "chest"):
            targets.append(geom)
    pairs = []
    for hand, gloves in enumerate(hands):
        selected = [
            (glove, target)
            for glove in gloves
            for target in targets
            if (model.geom_contype[glove] & model.geom_conaffinity[target])
            or (model.geom_contype[target] & model.geom_conaffinity[glove])
        ]
        if not selected:
            side = ("left", "right")[hand]
            raise ValueError(
                f"Punch reach: red {side} hand has no collidable official "
                "hand-to-opponent torso/chest pair"
            )
        pairs.append(selected)
    return torso_ids, pairs


def _velocity(model, data, kind, index):
    value = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, kind, index, value, 0)
    return value


def _point_velocity(model, data, geom, point):
    velocity = _velocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom)
    return velocity[3:] + np.cross(velocity[:3], point - data.geom_xpos[geom])


def _direction(*candidates):
    for candidate in candidates:
        length = np.linalg.norm(candidate)
        if length > 1e-10:
            return candidate / length
    return np.zeros(3, dtype=np.float64)


def _validate_distance_support(model, pairs):
    # Validate configuration on scratch FK, never on a live training world.
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    for hand in pairs:
        for glove, target in hand:
            points = np.zeros(6, dtype=np.float64)
            gap = mujoco.mj_geomDistance(
                model, data, glove, target, np.inf, points
            )
            if not np.isfinite(gap) or not np.isfinite(points).all():
                raise ValueError(
                    "Punch reach has unsupported/nonfinite model geom "
                    f"distance: {glove}, {target}"
                )


def _measure_world(model, data, torso_ids, pairs):
    values = np.empty((2, len(_VALUES)))
    torso = torso_ids[0]
    # XBODY measures the link origin. BODY would measure the inertial frame,
    # incorrectly mixing a COM velocity with an offset from the link origin.
    torso_velocity = _velocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, torso)
    for hand in range(2):
        best = None
        for glove, target in pairs[hand]:
            points = np.zeros(6, dtype=np.float64)
            gap = mujoco.mj_geomDistance(
                model, data, glove, target, np.inf, points
            )
            if not np.isfinite(gap) or not np.isfinite(points).all():
                return None
            if best is None or gap < best[0]:
                best = gap, glove, target, points.copy()
        gap, glove, target, points = best
        glove_point, target_point = points[:3], points[3:]
        centers = data.geom_xpos[target] - data.geom_xpos[glove]
        normal = _direction(
            centers if gap < 0 else target_point - glove_point,
            centers,
            data.xpos[torso_ids[1]] - data.xpos[torso],
        )
        glove_velocity = _point_velocity(model, data, glove, glove_point)
        target_velocity = _point_velocity(model, data, target, target_point)
        offset = glove_point - data.xpos[torso]
        arm_velocity = (
            glove_velocity
            - torso_velocity[3:]
            - np.cross(torso_velocity[:3], offset)
        )
        row = (
            gap,
            np.dot(glove_velocity - target_velocity, normal),
            np.dot(arm_velocity, normal),
            np.dot(arm_velocity, _direction(offset)),
        )
        if (
            not np.isfinite(row).all()
            or (np.abs(row) > np.finfo(np.float32).max).any()
        ):
            return None
        values[hand] = row
    return values


class PunchReachGeometry:
    """Measure red left/right hands as ``[worlds, 2]`` tensors.

    ``gap_m`` is the signed minimum surface separation for an officially
    recognized hand and a collidable opponent torso/chest. Head and pelvis
    are excluded. Positive closing speed includes both fighters' motion;
    positive arm speed removes the red torso's rigid translation/rotation.
    During overlap the center direction approximates the surface normal.

    Invalid runtime worlds retain NaNs with ``target_valid=False`` for the
    caller's normal failure/recovery path. Configuration errors are raised.
    Missing geometry is never silently replaced by a wrist/body-origin proxy.
    """

    def __init__(self, model):
        self._reference_bindings = _bindings(model)
        _validate_distance_support(model, self._reference_bindings[1])
        self._geom_types = model.geom_type.copy()
        self._models = {id(model): (model, self._reference_bindings)}

    def _model_bindings(self, model):
        key = id(model)
        if key not in self._models:
            bindings = _bindings(model)
            if bindings != self._reference_bindings or not np.array_equal(
                model.geom_type, self._geom_types
            ):
                raise ValueError("Punch reach world model layout differs")
            # Keep strong references: randomized models must not reuse IDs.
            self._models[key] = (model, bindings)
        return self._models[key][1]

    def measure(self, combat):
        physics = combat.physics
        worlds = len(physics.models)
        if worlds < 1 or len(physics.fk) != worlds:
            raise ValueError("Punch reach requires one FK state per world")
        values = np.full((worlds, 2, len(_VALUES)), np.nan)
        valid = np.zeros((worlds, 2), dtype=bool)
        for world, (model, data) in enumerate(
            zip(physics.models, physics.fk, strict=True)
        ):
            torso_ids, pairs = self._model_bindings(model)
            if not all(
                np.isfinite(getattr(data, name)).all() for name in _FK_FIELDS
            ):
                continue
            measured = _measure_world(model, data, torso_ids, pairs)
            if measured is not None:
                values[world] = measured
                valid[world] = True
        # Transfer all scalar measurements together rather than per field.
        packed = torch.as_tensor(
            values, dtype=torch.float32, device=combat.device
        )
        result = {key: packed[..., index] for index, key in enumerate(_VALUES)}
        result["target_valid"] = torch.as_tensor(
            valid, dtype=torch.bool, device=combat.device
        )
        return result
