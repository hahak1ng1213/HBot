"""Exact evaluator geom semantics shared with native training contacts."""

import mujoco

STRIKE_ALIASES = {
    "left_wrist_yaw_link": "left_hand",
    "right_wrist_yaw_link": "right_hand",
    "left_rubber_hand": "left_hand",
    "right_rubber_hand": "right_hand",
    "left_ankle_roll_link": "left_foot",
    "right_ankle_roll_link": "right_foot",
}


def strike_geometry(model, geom):
    """Match the evaluator's exact links, not arbitrary wrist substrings."""
    body = model.body(int(model.geom_bodyid[geom])).name or ""
    for fighter, prefix in enumerate(("p1_", "p2_")):
        if body.startswith(prefix):
            alias = STRIKE_ALIASES.get(body[len(prefix) :])
            if alias is not None:
                return fighter, alias
    return None


def opponent_target_semantic(model, attacker, geom):
    """Preserve body and attached-mesh overrides from the official scene."""
    body_id = int(model.geom_bodyid[geom])
    body = model.body(body_id).name or ""
    prefix = ("p1_", "p2_")[1 - attacker]
    if not body.startswith(prefix):
        return None
    link = body[len(prefix) :]
    semantic = {
        "torso_link": "torso",
        "pelvis": "pelvis",
        "head_link": "head",
        "logo_link": "chest",
    }.get(link, "unscored")
    mesh_id = (
        int(model.geom_dataid[geom])
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        else -1
    )
    mesh_name = (model.mesh(mesh_id).name or "") if mesh_id >= 0 else ""
    if mesh_name.startswith(prefix):
        mesh_name = mesh_name[len(prefix) :]
    semantic = {
        "head_link": "head",
        "logo_link": "chest",
        "pelvis_contour_link": "pelvis",
        "pelvis_contour_collision": "pelvis",
    }.get(mesh_name, semantic)
    return body_id, link, semantic
