"""Free-punch contracts and native synthetic physics; not skill acceptance."""

from types import SimpleNamespace

import pytest
import torch

from test_boxing_cpu_physics import physics as native_physics  # noqa: F401
from test_fixed_punch_task import env as base_env, place
from holoboxing.training.skills import fixed_punch
from holoboxing.training.skills.actor import SkillActor
from holoboxing.training.skills.free_punch import (
    FreePunchCombatEnv,
    PunchCycles,
    PaceCurriculum,
)
from holoboxing.training.skills.frozen_v6 import FrozenV6Opponent
from holoboxing.training.skills.train import train_teacher
from holoboxing.training.skills.train_free_punch import evaluate_free

env = base_env


def test_cycle_needs_outward_extension_and_recovery_and_attributes_correct_hand():
    c = PunchCycles(1, "cpu")
    active = torch.tensor([True])
    hits = torch.zeros(1, 2)

    def step(d, h=hits):
        p = torch.tensor([d], dtype=torch.float)
        return c.advance(p, p, h, active, active, 0.02)

    step([0, 0])
    for _ in range(20):
        assert not step([0, 0], torch.ones(1, 2))[1].any()  # parked contact
    step([0.18, 0])
    step([0.3, 0], torch.tensor([[1.0, 0.0]]))
    for _ in range(15):
        step([0.3, 0])
    complete, hit, duration, _ = step([0, 0])
    assert complete.tolist() == hit.tolist() == [[True, False]]
    assert duration[0, 0] >= 0.25
    assert not step([0, 0])[1].any()  # cannot double count a cycle
    step([0.3, 0])
    c.reset(torch.tensor([True]))
    assert not c.phase.any() and not c.hit.any()


def test_air_cycles_never_receive_hit_recovery_credit_and_timeout_clears_hit():
    c = PunchCycles(1, "cpu")
    on = torch.ones(1, dtype=torch.bool)
    for dist in [0, 0.3] + [0.3] * 20 + [0]:
        d = torch.full((1, 2), dist)
        result = c.advance(d, d, d * 0, on, on, 0.02)
    assert result[0].all() and not result[1].any()
    assert (c.interval > 0).all()  # air cycles cannot improve hit frequency
    c.phase[:] = 3
    c.hit[:] = True
    c.age[:] = 3.1
    result = c.advance(d, d, d * 0, on, on, 0.02)
    assert not result[1].any() and not c.hit.any()


def test_curriculum_waits_for_effective_cycles_and_demotes_when_unstable():
    c = PaceCurriculum()
    metrics = dict(
        hit_complete_cycles=0.0, complete_cycles=1.0, low_pose=0.0, invalid=0.0
    )
    for i in range(64):
        c.observe(i, metrics, 256)
    assert c.stage == 0  # thousands of air punches do not promote
    metrics.update(
        hit_complete_cycles=8 / (64 * 256), complete_cycles=12 / (64 * 256)
    )
    for i in range(64):
        c.observe(i, metrics, 256)
    assert c.stage == 1 and c.duration == 1.2
    metrics["low_pose"] = 0.04
    for i in range(64):
        c.observe(i, metrics, 256)
    assert c.stage == 0


def configure(env):
    torch.set_num_threads(1)
    env.combat.standing_reference = None
    mentor = SkillActor(1356).eval().requires_grad_(False)
    with torch.no_grad():
        mentor.actor[-1].bias.fill_(0.1)
    saved = dict(
        environment_recipe=env.recipe,
        reference_manifest=env.library.manifest,
        seed=37,
    )
    blue = FrozenV6Opponent(env, mentor, saved, fixed_punch, {"fixture": True})
    env.__class__ = FreePunchCombatEnv
    env.configure(mentor, blue, {"fixture": True}, 4)
    return env


def test_zero_residual_preserves_prior_targets_and_crouch_leaves_feet(env):
    place(env, 0.55)
    configure(env)
    raw = env.mentor.act(
        fixed_punch.FixedPunchCombatEnv.get_observations(env),
        deterministic=True,
    )
    expected = fixed_punch.FixedPunchCombatEnv.action_targets(env, raw)
    actual = env.action_targets(raw)
    torch.testing.assert_close(expected[2], actual[2])
    torch.testing.assert_close(expected[3], actual[3])
    env.learner_action[:, 2] = 1
    lowered = env.action_targets(raw)[2]
    torch.testing.assert_close(
        lowered[:, :, 0, 2], expected[2][:, :, 0, 2] - 0.08
    )
    torch.testing.assert_close(lowered[:, :, 3:], expected[2][:, :, 3:])
    env.learner_action[:, 6] = 1
    extended = env.action_targets(raw)[2]
    assert (extended[:, :, 1, 0] - lowered[:, :, 1, 0] > 0.39).all()


def test_native_free_ppo_freezes_blue_and_serializes_curriculum(env, tmp_path):
    torch.set_num_threads(1)
    configure(env)
    before = {k: v.clone() for k, v in env.mentor.state_dict().items()}
    assert env.observation_dim == 1397
    args = SimpleNamespace(
        num_envs=2,
        updates=2,
        rollout_steps=4,
        save_every=1,
        learning_rate=1e-5,
        seed=37,
        skill="free_punch_v7",
    )
    result = train_teacher(env, tmp_path / "train", args)
    for k, v in env.mentor.state_dict().items():
        torch.testing.assert_close(v, before[k])
    saved = torch.load(result["checkpoint"], weights_only=False)
    assert saved["format"] == "kangning037_free_punch_v7"
    assert len(saved["curriculum_state"]["window"]) == 2
    args.mode, args.checkpoint, args.output, args.eval_steps = (
        "evaluate",
        result["checkpoint"],
        tmp_path / "eval",
        2,
    )
    report = evaluate_free(env, args)
    assert report["steps"] == 2
    assert not env.opponent.pending
    assert all(not p.requires_grad for p in env.mentor.parameters())


def test_pending_blue_cannot_advance_physics_twice(env):
    configure(env)
    steps = env.physics.steps
    env.opponent(env.obs["robot_B"]["task_policy"])
    assert env.physics.steps == steps
    with pytest.raises(RuntimeError, match="one transition"):
        env.opponent(env.obs["robot_B"]["task_policy"])


def test_frozen_split_step_matches_full_teacher_targets_and_state(env):
    import copy
    from holoboxing.training.skills.frozen_v6 import swapped

    configure(env)
    env.reset_training_rows(torch.arange(env.num_envs))
    teacher = env.opponent.teacher
    submitted = {}
    original_submit = teacher.combat.step

    def capture(actions):
        submitted["actual"] = actions["robot_A"].clone()
        return original_submit(actions)

    teacher.combat.step = capture
    for _ in range(8):
        oracle = copy.copy(teacher)
        for key, value in teacher.__dict__.items():
            if torch.is_tensor(value):
                setattr(oracle, key, value.clone())
        before = {k: v.flip(1).clone() for k, v in env.physics.state().items()}
        replay = {"state": before}
        oracle.physics = SimpleNamespace(state=lambda: replay["state"])
        with torch.no_grad():
            raw = env.mentor.act(
                teacher.get_observations(), deterministic=True
            )
            _, _, done, info = env.step(torch.zeros(env.num_envs, 30))
        after = {k: v.flip(1).clone() for k, v in env.physics.state().items()}
        transition = {
            k: v.flip(1).clone()
            for k, v in env.combat.last_transition_state.items()
        }

        def submit(actions):
            submitted["oracle"] = actions["robot_A"].clone()
            replay["state"] = after
            return (
                swapped(env.obs),
                None,
                done,
                {"invalid_envs": info["metrics"]["invalid"].bool()},
            )

        oracle.combat = SimpleNamespace(
            step=submit,
            metrics=swapped(env.combat.metrics),
            last_transition_state=transition,
        )
        oracle.step(raw)
        torch.testing.assert_close(submitted["actual"], submitted["oracle"])
        for name in (
            "phase",
            "age",
            "attack",
            "punching",
            "anchor",
            "heading",
            "entry_p",
            "entry_q",
            "last_action",
            "no_torso_contact_steps",
        ):
            torch.testing.assert_close(
                getattr(teacher, name), getattr(oracle, name)
            )


def test_official_hit_is_not_discarded_by_range_gate_and_invalid_is_failure(
    env, monkeypatch
):
    place(env, 1.2)
    configure(env)
    assert not env.punching.any()
    original = env.combat.step

    def scored_step(actions):
        obs, reward, done, info = original(actions)
        env.combat.metrics["robot_A"]["official_hits"][:] = 1
        env.combat.metrics["robot_A"]["official_left_hits"][:] = 1
        info["invalid_envs"][1] = True
        done[1] = True
        return obs, reward, done, info

    monkeypatch.setattr(env.combat, "step", scored_step)
    _, reward, _, info = env.step(torch.zeros(2, 30))
    assert info["metrics"]["official_hit"].tolist() == [10.0, 0.0]
    assert reward[1] == -10
    assert info["failed"][1]
    assert not info["metrics"]["hit_complete_cycles"].any()
